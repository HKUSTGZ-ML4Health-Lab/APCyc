"""Bounded forward/backward and serialization checks; never call optimizer.step()."""
from copy import deepcopy
from itertools import islice
import json
from pathlib import Path

import torch

from utils.checkpoint_utils import load_model_checkpoint, unwrap_model


def gradient_summary(model):
    grads = [p.grad.detach() for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(torch.isfinite(g).all().item() for g in grads):
        raise RuntimeError('Preflight requires finite gradients on trainable parameters.')
    norm = sum(g.double().square().sum().item() for g in grads) ** 0.5
    if norm == 0:
        raise RuntimeError('Preflight found only zero gradients.')
    return {'gradient_tensors': len(grads), 'gradient_norm': norm}


def verify_state_unchanged(before, model):
    after = unwrap_model(model).state_dict()
    if set(before) != set(after) or any(not torch.equal(v, after[k].detach().cpu()) for k, v in before.items()):
        raise RuntimeError('Model state changed during a zero-update preflight.')


def run_trainer_preflight(trainer, device_ids, local_rank, batches, out_dir):
    from torch.utils.tensorboard import SummaryWriter
    distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    world_size = torch.distributed.get_world_size() if distributed else 1
    root = Path(out_dir)
    output = root / f'rank{rank}'
    output.mkdir(parents=True, exist_ok=False)
    device_id = local_rank if distributed else device_ids[0]
    device = torch.device('cpu' if device_id == -1 else f'cuda:{device_id}')
    trainer.local_rank = local_rank
    trainer.writer = SummaryWriter(str(output / 'tensorboard')) if trainer._is_main_proc() else None
    trainer.model.to(device).train()
    before = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
    if distributed:
        trainer.model = torch.nn.parallel.DistributedDataParallel(
            trainer.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    checks = []
    last_batch = None
    for step, batch in enumerate(islice(trainer.train_loader, batches)):
        batch = trainer.to_device(batch, device)
        trainer.model.zero_grad(set_to_none=True)
        loss = trainer.train_step(deepcopy(batch), step)
        if trainer.is_oom_return(loss) or not torch.isfinite(loss).all().item():
            raise RuntimeError('Nonfinite loss or OOM during preflight.')
        loss.backward()
        grad = gradient_summary(trainer.model)
        if distributed:
            norm = torch.tensor(grad['gradient_norm'], dtype=torch.float64, device=device)
            norms = [torch.empty_like(norm) for _ in range(world_size)]
            torch.distributed.all_gather(norms, norm)
            if not all(torch.allclose(norm, other, rtol=1e-8, atol=1e-10) for other in norms):
                raise RuntimeError('Distributed gradients disagree across ranks.')
        checks.append({'batch': step, 'loss': loss.item(), 'samples': int(batch['lengths'].numel()), **grad})
        last_batch = batch
    if len(checks) != batches:
        raise ValueError(f'Preflight requested {batches} batches but this rank only has {len(checks)}.')
    verify_state_unchanged(before, trainer.model)
    checkpoint = output / 'model.ckpt'
    trainer.save_checkpoint(checkpoint, preflight=True)
    model = unwrap_model(trainer.model)
    if type(model).__name__ == 'APCycLDMPepDesign':
        from apcyc.checkpoint import load_bundle
        restored, _ = load_bundle(checkpoint, 'cpu')
    else:
        restored, _ = load_model_checkpoint(checkpoint, 'cpu')
    verify_state_unchanged(before, restored)
    # GPU scatter reductions are not bitwise deterministic. Check the serializer
    # with equal CPU inputs while retaining the real GPU backward checks above.
    model.to('cpu').eval()
    restored.eval()
    last_batch = trainer.to_device(deepcopy(last_batch), torch.device('cpu'))
    losses = []
    for candidate in [model, restored]:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(2021)
            result = candidate(**deepcopy(last_batch))
            losses.append(result[0].detach())
    if not torch.isfinite(torch.stack(losses)).all() or not torch.allclose(*losses, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f'Reloaded model output differs: {[x.item() for x in losses]}')
    report = dict(model=type(model).__name__, rank=rank, world_size=world_size, device=str(device),
                  optimizer_steps=0, weights_unchanged=True, exact_state_reload=True,
                  reload_output_comparison_device='cpu',
                  reloaded_losses=[x.item() for x in losses], batches=checks, checkpoint=str(checkpoint))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if trainer.writer:
        trainer.writer.close()
    if distributed:
        reports = [None] * world_size
        torch.distributed.all_gather_object(reports, report)
    else:
        reports = [report]
    if rank == 0:
        (root / 'summary.json').write_text(json.dumps({'optimizer_steps': 0, 'ranks': reports}, indent=2) + '\n')
        print(f'Preflight passed on {world_size} rank(s); zero optimizer updates. Results: {root}', flush=True)
