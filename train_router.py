#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import datetime
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from router.data import RouterDataset, build_router_mmap, router_collate
from router.model import RouterModel
from router.pairs import ValidPairGenerator
from utils.logger import print_log


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_list(s: str, dtype=float) -> List:
    if not s:
        return []
    return [dtype(x) for x in s.split(",")]


def parse_topk(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x]

def format_metrics(metrics: Dict[str, float], type_list: List[str], topk_list: List[int]) -> str:
    parts = []
    for key in ("loss", "type_acc", "type_macro_f1"):
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    for idx, t in enumerate(type_list):
        key = f"type_recall_{idx}"
        if key in metrics:
            parts.append(f"type_recall_{t}={metrics[key]:.4f}")
    for k in topk_list:
        key = f"site_top{k}"
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    for k in topk_list:
        key = f"joint_top{k}"
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    for t in ("SS", "ISO"):
        for k in (1, 3, 5):
            key = f"pair_hit{k}_{t}"
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")
    return " ".join(parts)

def flatten_config(cfg: Dict) -> Dict:
    flat: Dict = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                flat[sub_key] = sub_val
        else:
            flat[key] = val
    return flat

def apply_config(args, cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    flat = flatten_config(cfg)
    for key, val in flat.items():
        if hasattr(args, key):
            setattr(args, key, val)
        else:
            print_log(f"Config key ignored (unknown): {key}")

def create_scheduler(optimizer, args, monitor_name: str, min_delta: float):
    if not args.use_scheduler:
        return None, monitor_name
    if args.scheduler_monitor not in (None, "auto"):
        monitor_name = args.scheduler_monitor
    if args.scheduler_class != "ReduceLROnPlateau":
        raise ValueError(f"Unsupported scheduler: {args.scheduler_class}")
    if args.scheduler_mode == "auto":
        mode = "min" if "loss" in monitor_name else "max"
    else:
        mode = args.scheduler_mode
    threshold = args.scheduler_threshold if args.scheduler_threshold is not None else min_delta
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        mode=mode,
        threshold=threshold,
        cooldown=args.scheduler_cooldown,
        min_lr=args.scheduler_min_lr,
        verbose=bool(args.scheduler_verbose),
    )
    return scheduler, monitor_name

def parse_type_map(spec: str, type_list: List[str]) -> Optional[List[float]]:
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        return None
    mapping: Dict[str, float] = {}
    if all(":" in p for p in parts):
        for item in parts:
            key, val = item.split(":")
            mapping[key.strip()] = float(val)
        weights = []
        for t in type_list:
            if t not in mapping:
                raise ValueError(f"Missing weight for type {t} in {spec}")
            weights.append(mapping[t])
        return weights
    if len(parts) != len(type_list):
        raise ValueError(f"Expected {len(type_list)} weights, got {len(parts)} in {spec}")
    return [float(v) for v in parts]


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(n * val_ratio)
    return indices[split:], indices[:split]

def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def load_router_ckpt(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt if isinstance(ckpt, dict) else {}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print_log(f"Resume router: missing={len(missing)} unexpected={len(unexpected)}")


def load_id_list(path: str) -> List[str]:
    ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0]
            token = os.path.basename(token)
            if token.lower().endswith(".pdb"):
                token = token[:-4]
            ids.append(token)
    return ids

@dataclass
class EarlyStopConfig:
    monitor: str
    patience: int
    min_delta: float
    max_epochs: int


class EarlyStopping:
    def __init__(self, cfg: EarlyStopConfig) -> None:
        self.cfg = cfg
        self.best: Optional[float] = None
        self.bad_epochs = 0
        self.epoch = 0

    def step(self, metric: float) -> bool:
        self.epoch += 1
        if self.best is None or metric > self.best + self.cfg.min_delta:
            self.best = metric
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    def should_stop(self) -> bool:
        if self.epoch >= self.cfg.max_epochs:
            return True
        return self.bad_epochs > self.cfg.patience


def compute_type_losses(
    logits_c: torch.Tensor,
    c_star: torch.Tensor,
    type_soft_list: List[torch.Tensor],
    num_types: int,
    use_hard: bool,
    use_soft: bool,
    soft_weight: float,
    ce_weight: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce = None
    metrics = {}
    if use_hard:
        weight = None if ce_weight is None else ce_weight.to(logits_c.device)
        ce = F.cross_entropy(logits_c, c_star, weight=weight)
        metrics["type_ce"] = float(ce.item())
    if not use_soft:
        if ce is None:
            raise ValueError("type loss has no hard term and soft term disabled.")
        return ce, metrics

    soft_rows = []
    mask = []
    for ts in type_soft_list:
        if ts.numel() == num_types:
            soft_rows.append(ts)
            mask.append(True)
        else:
            soft_rows.append(torch.zeros(num_types))
            mask.append(False)
    if not any(mask):
        if ce is None:
            raise ValueError("type soft loss requested but no soft labels found.")
        return ce, metrics
    soft = torch.stack(soft_rows).to(logits_c.device)
    mask_t = torch.tensor(mask, device=logits_c.device, dtype=torch.bool)
    soft = soft[mask_t]
    soft = soft / soft.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    log_probs = F.log_softmax(logits_c[mask_t], dim=-1)
    kl = F.kl_div(log_probs, soft, reduction="batchmean")
    metrics["type_kl"] = float(kl.item())
    if ce is None:
        return soft_weight * kl, metrics
    return ce + soft_weight * kl, metrics


def _pair_soft_weights(
    pair_soft: torch.Tensor,
    pair_index: Dict[Tuple[int, int], int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if pair_soft.numel() == 0:
        return None
    weights = torch.zeros(len(pair_index), device=device)
    for i, j, w in pair_soft.tolist():
        idx = pair_index.get((int(i), int(j)))
        if idx is None:
            continue
        weights[idx] = float(w)
    if weights.sum() <= 0:
        return None
    weights = weights / weights.sum().clamp(min=1e-6)
    return weights


def compute_site_loss(
    logits: torch.Tensor,
    pairs: List[Tuple[int, int]],
    gt_pair: torch.Tensor,
    pair_soft: torch.Tensor,
    use_hard: bool,
    use_soft: bool,
    soft_weight: float,
) -> Optional[torch.Tensor]:
    if not pairs or logits.numel() == 0:
        return None
    log_probs = F.log_softmax(logits, dim=-1)
    pair_index = {pair: idx for idx, pair in enumerate(pairs)}

    hard_loss = None
    if use_hard:
        gt = (int(gt_pair[0].item()), int(gt_pair[1].item()))
        if gt not in pair_index:
            return None
        hard_loss = -log_probs[pair_index[gt]]

    if not use_soft:
        return hard_loss
    soft_weights = _pair_soft_weights(pair_soft, pair_index, logits.device)
    if soft_weights is None:
        return hard_loss
    soft_loss = -(soft_weights * log_probs).sum()
    if hard_loss is None:
        return soft_weight * soft_loss
    return hard_loss + soft_weight * soft_loss


def compute_site_topk(
    logits: torch.Tensor,
    pairs: List[Tuple[int, int]],
    gt_pair: torch.Tensor,
    topk_list: List[int],
) -> Dict[int, bool]:
    results = {k: False for k in topk_list}
    if not pairs or logits.numel() == 0:
        return results
    gt = (int(gt_pair[0].item()), int(gt_pair[1].item()))
    pair_index = {pair: idx for idx, pair in enumerate(pairs)}
    if gt not in pair_index:
        return results
    idx = pair_index[gt]
    for k in topk_list:
        k = min(k, logits.numel())
        topk = logits.topk(k).indices
        results[k] = int(idx in topk.tolist()) == 1
    return results


def run_epoch(
    model: RouterModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    pair_gen: ValidPairGenerator,
    topk_list: List[int],
    lambda1: float,
    lambda2: float,
    use_type_hard: bool,
    use_type_soft: bool,
    type_soft_weight: float,
    type_ce_weight: Optional[torch.Tensor],
    use_pair_hard: bool,
    use_pair_soft: bool,
    pair_soft_weight: float,
    site_condition: str,
    wrong_type_weight: float,
    pred_prob: float,
    gradient_clip: float,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0
    type_correct = 0
    type_total = 0
    type_recall = {i: [0, 0] for i in range(model.num_types)}
    type_tp = [0 for _ in range(model.num_types)]
    type_fp = [0 for _ in range(model.num_types)]
    type_fn = [0 for _ in range(model.num_types)]

    site_correct = {k: 0 for k in topk_list}
    site_total = 0
    joint_correct = {k: 0 for k in topk_list}
    joint_total = 0
    pair_eval_k = [1, 3, 5]
    pair_eval_types = {t for t in model.type_list if t in ("SS", "ISO")}
    pair_eval_hits = {t: {k: 0 for k in pair_eval_k} for t in pair_eval_types}
    pair_eval_total = {t: 0 for t in pair_eval_types}

    for batch in loader:
        if batch is None:
            continue
        batch_graph, labels = batch
        batch_graph = batch_graph.to(device)
        c_star = labels["c_star"].to(device)
        peptide_lengths = labels["peptide_length"]

        hb_nodes, hb = model.encode_pocket(batch_graph)
        logits_c = model.predict_type(hb)
        c_pred = logits_c.argmax(dim=-1)

        type_loss, _ = compute_type_losses(
            logits_c,
            c_star,
            labels["type_soft"],
            model.num_types,
            use_type_hard,
            use_type_soft,
            type_soft_weight,
            type_ce_weight,
        )

        loss = lambda1 * type_loss

        if site_condition != "none":
            if site_condition == "gt":
                c_cond = c_star.tolist()
                pred_mask = [False] * len(c_cond)
            elif site_condition == "pred":
                c_cond = c_pred.tolist()
                pred_mask = [True] * len(c_cond)
            elif site_condition == "mix":
                c_cond = []
                pred_mask = []
                for i in range(len(c_star)):
                    if random.random() < pred_prob:
                        c_cond.append(int(c_pred[i].item()))
                        pred_mask.append(True)
                    else:
                        c_cond.append(int(c_star[i].item()))
                        pred_mask.append(False)
            else:
                raise ValueError(f"Unknown site_condition: {site_condition}")

            scores_for_loss = model.predict_site_logits(
                hb_nodes, batch_graph, peptide_lengths, c_cond, pair_gen
            )
            scores_for_metrics = model.predict_site_logits(
                hb_nodes, batch_graph, peptide_lengths, c_star.tolist(), pair_gen
            )

            site_losses = []
            site_weights = []
            for i, (pairs, logits) in enumerate(scores_for_loss):
                gt_pair = labels["gt_pair"][i]
                pair_soft = labels["pair_soft"][i]

                sample_loss = compute_site_loss(
                    logits,
                    pairs,
                    gt_pair,
                    pair_soft,
                    use_pair_hard,
                    use_pair_soft,
                    pair_soft_weight,
                )
                if sample_loss is None:
                    continue
                if pred_mask[i]:
                    weight = 1.0 if c_pred[i] == c_star[i] else wrong_type_weight
                else:
                    weight = 1.0
                site_losses.append(sample_loss * weight)
                site_weights.append(weight)

            if site_losses:
                site_loss = torch.stack(site_losses).sum() / max(
                    sum(site_weights), 1.0
                )
                loss = loss + lambda2 * site_loss

            for i, (pairs, logits) in enumerate(scores_for_metrics):
                gt_pair = labels["gt_pair"][i]
                topk_hits = compute_site_topk(logits, pairs, gt_pair, topk_list)
                if int(c_pred[i].item()) == int(c_star[i].item()):
                    site_total += 1
                    for k in topk_list:
                        site_correct[k] += int(topk_hits[k])
                joint_total += 1
                for k in topk_list:
                    joint_correct[k] += int(
                        topk_hits[k] and (int(c_pred[i].item()) == int(c_star[i].item()))
                    )
                type_name = model.type_list[int(c_star[i].item())]
                if type_name in pair_eval_types:
                    pair_hits = compute_site_topk(logits, pairs, gt_pair, pair_eval_k)
                    pair_eval_total[type_name] += 1
                    for k in pair_eval_k:
                        pair_eval_hits[type_name][k] += int(pair_hits[k])

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if gradient_clip and gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        type_total += c_star.numel()
        type_correct += int((c_pred == c_star).sum().item())
        for cls in range(model.num_types):
            mask = c_star == cls
            if mask.any():
                type_recall[cls][1] += int(mask.sum().item())
                type_recall[cls][0] += int((c_pred[mask] == cls).sum().item())
            pred_mask = c_pred == cls
            if pred_mask.any():
                type_fp[cls] += int((pred_mask & (c_star != cls)).sum().item())
            if mask.any():
                type_fn[cls] += int((~pred_mask & mask).sum().item())
            if pred_mask.any() and mask.any():
                type_tp[cls] += int((pred_mask & mask).sum().item())

    metrics = {}
    metrics["loss"] = total_loss / max(total_batches, 1)
    metrics["type_acc"] = type_correct / max(type_total, 1)
    for cls in range(model.num_types):
        correct, total = type_recall[cls]
        metrics[f"type_recall_{cls}"] = correct / max(total, 1)
    f1_list = []
    for cls in range(model.num_types):
        tp = type_tp[cls]
        fp = type_fp[cls]
        fn = type_fn[cls]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        denom = precision + recall
        f1 = 0.0 if denom == 0 else 2 * precision * recall / denom
        f1_list.append(f1)
    metrics["type_macro_f1"] = sum(f1_list) / max(len(f1_list), 1)

    for k in topk_list:
        metrics[f"site_top{k}"] = site_correct[k] / max(site_total, 1)
        metrics[f"joint_top{k}"] = joint_correct[k] / max(joint_total, 1)
    for t in sorted(pair_eval_types):
        total = max(pair_eval_total[t], 1)
        for k in pair_eval_k:
            metrics[f"pair_hit{k}_{t}"] = pair_eval_hits[t][k] / total

    return metrics


def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def train_stage(
    stage_name: str,
    model: RouterModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pair_gen: ValidPairGenerator,
    topk_list: List[int],
    lambda1: float,
    lambda2: float,
    use_type_hard: bool,
    use_type_soft: bool,
    type_soft_weight: float,
    type_ce_weight: Optional[torch.Tensor],
    use_pair_hard: bool,
    use_pair_soft: bool,
    pair_soft_weight: float,
    site_condition: str,
    wrong_type_weight: float,
    pred_prob: float,
    gradient_clip: float,
    scheduler_args,
    early_cfg: EarlyStopConfig,
    output_dir: str,
    writer: Optional[SummaryWriter],
) -> Dict[str, float]:
    stopper = EarlyStopping(early_cfg)
    best_metric = -1e9
    best_path = os.path.join(output_dir, f"{stage_name}_best.pt")
    best_metrics: Dict[str, float] = {}
    scheduler, scheduler_monitor = create_scheduler(
        optimizer, scheduler_args, early_cfg.monitor, early_cfg.min_delta
    )

    while True:
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pair_gen,
            topk_list,
            lambda1,
            lambda2,
            use_type_hard,
            use_type_soft,
            type_soft_weight,
            type_ce_weight,
            use_pair_hard,
            use_pair_soft,
            pair_soft_weight,
            site_condition,
            wrong_type_weight,
            pred_prob,
            gradient_clip,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            pair_gen,
            topk_list,
            lambda1,
            lambda2,
            use_type_hard,
            use_type_soft,
            type_soft_weight,
            type_ce_weight,
            use_pair_hard,
            use_pair_soft,
            pair_soft_weight,
            "gt",
            wrong_type_weight,
            pred_prob,
            gradient_clip,
        )

        metric_value = val_metrics.get(early_cfg.monitor, 0.0)
        is_best = stopper.step(metric_value)
        if is_best:
            best_metric = metric_value
            save_checkpoint(model, best_path)
            best_metrics = dict(val_metrics)

        if writer is not None:
            epoch = stopper.epoch
            for key, val in train_metrics.items():
                writer.add_scalar(f"{stage_name}/train_{key}", val, epoch)
            for key, val in val_metrics.items():
                writer.add_scalar(f"{stage_name}/val_{key}", val, epoch)
            if scheduler is not None:
                writer.add_scalar(f"{stage_name}/lr", optimizer.param_groups[0]["lr"], epoch)

        if scheduler is not None:
            if scheduler_monitor in val_metrics:
                scheduler.step(val_metrics[scheduler_monitor])
            else:
                print_log(f"Warning: scheduler_monitor {scheduler_monitor} not found in val metrics.")

        print_log(
            f"{stage_name} epoch {stopper.epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_{early_cfg.monitor}={metric_value:.4f}"
        )
        detail = format_metrics(val_metrics, model.type_list, topk_list)
        if detail:
            print_log(f"{stage_name} epoch {stopper.epoch} val_metrics: {detail}")

        if stopper.should_stop():
            break

    print_log(f"{stage_name} best {early_cfg.monitor}={best_metric:.4f}")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    return {"best_metric": best_metric, "best_path": best_path, "best_metrics": best_metrics}


def run_training(
    args,
        train_set,
        val_set,
        device: torch.device,
        max_peptide_len: int,
) -> Dict[str, float]:
    model = RouterModel(
        type_list=args.type_list,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        encoder_layers=args.encoder_layers,
        encoder_dropout=args.encoder_dropout,
        channel_nf=args.channel_nf,
        radial_nf=args.radial_nf,
        max_position=args.max_position,
        backbone_only=args.backbone_only,
        edge_mode=args.edge_mode,
        edge_k=args.edge_k,
        edge_radius=args.edge_radius,
        max_peptide_len=max_peptide_len,
        site_heads=args.site_heads,
        site_dropout=args.site_dropout,
        type_condition=args.type_condition,
        use_cross_attention=not args.no_cross_attention,
    ).to(device)

    if args.resume_router_ckpt:
        load_router_ckpt(model, args.resume_router_ckpt)

    if args.pretrained_mode != "none":
        stats = model.encoder.load_pretrained(args.pretrained_ckpt, args.pretrained_mode)
        print_log(
            f"Pretrained load: loaded={stats['loaded']} missing={stats['missing']} mismatch={stats['mismatch']}"
        )

    frozen = model.encoder.freeze(args.freeze_mode, args.freeze_layers)
    if frozen:
        print_log(f"Frozen params: {len(frozen)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if isinstance(train_set, Subset):
        base_dataset = train_set.dataset
        train_indices = list(train_set.indices)
    else:
        base_dataset = train_set
        train_indices = list(range(len(train_set)))

    sampler = None
    if args.use_class_balanced_sampler:
        target = parse_type_map(args.sampler_target, args.type_list)
        if target is None:
            raise ValueError("sampler_target is required when use_class_balanced_sampler is set.")
        target_sum = sum(target)
        if target_sum <= 0:
            raise ValueError("sampler_target must have positive weights.")
        target = [v / target_sum for v in target]
        counts = [0 for _ in args.type_list]
        sample_weights = []
        for idx in train_indices:
            c_idx = base_dataset.get_c_star(idx)
            if c_idx < 0:
                sample_weights.append(0.0)
                continue
            counts[c_idx] += 1
        for idx in train_indices:
            c_idx = base_dataset.get_c_star(idx)
            if c_idx < 0 or counts[c_idx] == 0:
                sample_weights.append(0.0)
            else:
                sample_weights.append(target[c_idx] / counts[c_idx])
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=router_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=router_collate,
        drop_last=False,
    )

    pair_gen = ValidPairGenerator(
        args.type_list, rule=args.pair_rule, min_sep=args.pair_min_sep
    )
    writer = SummaryWriter(args.log_dir) if args.log_dir else None

    use_type_hard = True
    use_pair_hard = True
    use_type_soft = args.use_type_soft
    use_pair_soft = args.use_pair_soft
    if args.train_mode == "finetune":
        use_type_hard = False
        use_pair_hard = False
        use_type_soft = True
        use_pair_soft = True

    type_ce_weight = None
    if args.use_type_weighted_ce:
        weights = parse_type_map(args.type_class_weights, args.type_list)
        if weights is None:
            raise ValueError("type_class_weights is required when use_type_weighted_ce is set.")
        type_ce_weight = torch.tensor(weights, dtype=torch.float)

    results = {}
    if args.train_mode == "stage":
        stage1_cfg = EarlyStopConfig(
            monitor=args.stage1_monitor,
            patience=args.stage1_patience,
            min_delta=args.stage1_min_delta,
            max_epochs=args.stage1_max_epochs,
        )
        stage2_cfg = EarlyStopConfig(
            monitor=args.stage2_monitor,
            patience=args.stage2_patience,
            min_delta=args.stage2_min_delta,
            max_epochs=args.stage2_max_epochs,
        )

        results["stage1"] = train_stage(
            "stage1_type",
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            pair_gen,
            args.site_topk,
            lambda1=1.0,
            lambda2=0.0,
            use_type_hard=use_type_hard,
            use_type_soft=use_type_soft,
            type_soft_weight=args.type_soft_weight,
            type_ce_weight=type_ce_weight,
            use_pair_hard=use_pair_hard,
            use_pair_soft=use_pair_soft,
            pair_soft_weight=args.pair_soft_weight,
            site_condition="none",
            wrong_type_weight=args.joint_wrong_type_weight,
            pred_prob=args.stage3_pred_prob,
            gradient_clip=args.gradient_clip,
            scheduler_args=args,
            early_cfg=stage1_cfg,
            output_dir=args.output_dir,
            writer=writer,
        )

        results["stage2"] = train_stage(
            "stage2_site",
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            pair_gen,
            args.site_topk,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            use_type_hard=use_type_hard,
            use_type_soft=use_type_soft,
            type_soft_weight=args.type_soft_weight,
            type_ce_weight=type_ce_weight,
            use_pair_hard=use_pair_hard,
            use_pair_soft=use_pair_soft,
            pair_soft_weight=args.pair_soft_weight,
            site_condition="gt",
            wrong_type_weight=args.joint_wrong_type_weight,
            pred_prob=args.stage3_pred_prob,
            gradient_clip=args.gradient_clip,
            scheduler_args=args,
            early_cfg=stage2_cfg,
            output_dir=args.output_dir,
            writer=writer,
        )

        if args.enable_stage3:
            stage3_cfg = EarlyStopConfig(
                monitor=args.stage3_monitor,
                patience=args.stage3_patience,
                min_delta=args.stage3_min_delta,
                max_epochs=args.stage3_max_epochs,
            )
            results["stage3"] = train_stage(
                "stage3_joint",
                model,
                train_loader,
                val_loader,
                optimizer,
                device,
                pair_gen,
                args.site_topk,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                use_type_hard=use_type_hard,
                use_type_soft=use_type_soft,
                type_soft_weight=args.type_soft_weight,
                type_ce_weight=type_ce_weight,
                use_pair_hard=use_pair_hard,
                use_pair_soft=use_pair_soft,
                pair_soft_weight=args.pair_soft_weight,
                site_condition=args.stage3_type_condition,
                wrong_type_weight=args.joint_wrong_type_weight,
                pred_prob=args.stage3_pred_prob,
                gradient_clip=args.gradient_clip,
                scheduler_args=args,
                early_cfg=stage3_cfg,
                output_dir=args.output_dir,
                writer=writer,
            )

    elif args.train_mode == "stage3":
        if args.resume_router_ckpt is None:
            print_log("Warning: stage3 without resume_router_ckpt will train from scratch.")
        stage3_cfg = EarlyStopConfig(
            monitor=args.stage3_monitor,
            patience=args.stage3_patience,
            min_delta=args.stage3_min_delta,
            max_epochs=args.stage3_max_epochs,
        )
        results["stage3"] = train_stage(
            "stage3_joint",
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            pair_gen,
            args.site_topk,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            use_type_hard=use_type_hard,
            use_type_soft=use_type_soft,
            type_soft_weight=args.type_soft_weight,
            type_ce_weight=type_ce_weight,
            use_pair_hard=use_pair_hard,
            use_pair_soft=use_pair_soft,
            pair_soft_weight=args.pair_soft_weight,
            site_condition=args.stage3_type_condition,
            wrong_type_weight=args.joint_wrong_type_weight,
            pred_prob=args.stage3_pred_prob,
            gradient_clip=args.gradient_clip,
            scheduler_args=args,
            early_cfg=stage3_cfg,
            output_dir=args.output_dir,
            writer=writer,
        )

    elif args.train_mode in ("joint", "finetune"):
        joint_cfg = EarlyStopConfig(
            monitor=args.joint_monitor,
            patience=args.stage2_patience,
            min_delta=args.stage2_min_delta,
            max_epochs=args.stage2_max_epochs,
        )
        results["joint"] = train_stage(
            "joint",
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            pair_gen,
            args.site_topk,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            use_type_hard=use_type_hard,
            use_type_soft=use_type_soft,
            type_soft_weight=args.type_soft_weight,
            type_ce_weight=type_ce_weight,
            use_pair_hard=use_pair_hard,
            use_pair_soft=use_pair_soft,
            pair_soft_weight=args.pair_soft_weight,
            site_condition=args.joint_type_condition,
            wrong_type_weight=args.joint_wrong_type_weight,
            pred_prob=args.stage3_pred_prob,
            gradient_clip=args.gradient_clip,
            scheduler_args=args,
            early_cfg=joint_cfg,
            output_dir=args.output_dir,
            writer=writer,
        )
    else:
        raise ValueError(f"Unknown train_mode: {args.train_mode}")

    if writer is not None:
        writer.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cyclization router")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--type_list", type=str, default="HT,SS,ISO")
    parser.add_argument("--chain_id", type=str, default="R")
    parser.add_argument("--backbone_only", action="store_true")
    parser.add_argument("--use_mmap", action="store_true")
    parser.add_argument("--mmap_dir", type=str, default=None)
    parser.add_argument("--build_mmap", action="store_true")

    parser.add_argument("--train_mode", type=str, default="stage", choices=["stage", "joint", "finetune", "stage3"])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--train_ids", type=str, default=None)
    parser.add_argument("--valid_ids", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0)

    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--channel_nf", type=int, default=64)
    parser.add_argument("--radial_nf", type=int, default=128)
    parser.add_argument("--max_position", type=int, default=2048)
    parser.add_argument("--max_peptide_len", type=int, default=128)
    parser.add_argument("--edge_mode", type=str, default="knn", choices=["knn", "radius", "full"])
    parser.add_argument("--edge_k", type=int, default=16)
    parser.add_argument("--edge_radius", type=float, default=8.0)
    parser.add_argument("--site_heads", type=int, default=4)
    parser.add_argument("--site_dropout", type=float, default=0.1)
    parser.add_argument("--type_condition", type=str, default="add", choices=["add", "film"])
    parser.add_argument("--no_cross_attention", action="store_true")

    parser.add_argument("--pretrained_mode", type=str, default="none", choices=["none", "full", "partial"])
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--resume_router_ckpt", type=str, default=None)
    parser.add_argument("--freeze_mode", type=str, default="none", choices=["none", "all", "partial"])
    parser.add_argument("--freeze_layers", type=int, default=0)

    parser.add_argument("--use_type_soft", action="store_true")
    parser.add_argument("--type_soft_weight", type=float, default=0.5)
    parser.add_argument("--use_pair_soft", action="store_true")
    parser.add_argument("--pair_soft_weight", type=float, default=0.5)
    parser.add_argument("--use_type_weighted_ce", action="store_true")
    parser.add_argument("--type_class_weights", type=str, default="HT:1.69,ISO:0.636,SS:2.39")
    parser.add_argument("--use_class_balanced_sampler", action="store_true")
    parser.add_argument("--sampler_target", type=str, default="ISO:0.5,HT:0.25,SS:0.25")

    parser.add_argument("--pair_rule", type=str, default="default")
    parser.add_argument("--pair_min_sep", type=int, default=1)

    parser.add_argument("--joint_type_condition", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--joint_wrong_type_weight", type=float, default=0.0)

    parser.add_argument("--enable_stage3", action="store_true")
    parser.add_argument("--stage3_type_condition", type=str, default="pred", choices=["gt", "pred", "mix"])
    parser.add_argument("--stage3_pred_prob", type=float, default=0.5)

    parser.add_argument("--stage1_monitor", type=str, default="type_acc")
    parser.add_argument("--stage1_patience", type=int, default=2)
    parser.add_argument("--stage1_min_delta", type=float, default=0.002)
    parser.add_argument("--stage1_max_epochs", type=int, default=6)
    parser.add_argument("--stage2_monitor", type=str, default="site_top5")
    parser.add_argument("--stage2_patience", type=int, default=3)
    parser.add_argument("--stage2_min_delta", type=float, default=0.005)
    parser.add_argument("--stage2_max_epochs", type=int, default=10)
    parser.add_argument("--stage3_monitor", type=str, default="joint_top5")
    parser.add_argument("--stage3_patience", type=int, default=2)
    parser.add_argument("--stage3_min_delta", type=float, default=0.003)
    parser.add_argument("--stage3_max_epochs", type=int, default=4)
    parser.add_argument("--joint_monitor", type=str, default="joint_top5")

    parser.add_argument("--site_topk", type=str, default="1,5,10")
    parser.add_argument("--output_dir", type=str, default="./router_runs")
    parser.add_argument("--log_dir", type=str, default="./router_logs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default="./router_results")
    parser.add_argument("--result_name", type=str, default=None)

    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--scheduler_class", type=str, default="ReduceLROnPlateau")
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=1)
    parser.add_argument("--scheduler_mode", type=str, default="auto")
    parser.add_argument("--scheduler_threshold", type=float, default=None)
    parser.add_argument("--scheduler_cooldown", type=int, default=0)
    parser.add_argument("--scheduler_min_lr", type=float, default=1.0e-6)
    parser.add_argument("--scheduler_verbose", type=int, default=1)
    parser.add_argument("--scheduler_monitor", type=str, default="auto")

    parser.add_argument("--search", action="store_true")
    parser.add_argument("--search_lr", type=str, default="")
    parser.add_argument("--search_weight_decay", type=str, default="")
    parser.add_argument("--search_batch_size", type=str, default="")
    parser.add_argument("--search_metric", type=str, default="joint_top5")

    args = parser.parse_args()
    if args.config:
        apply_config(args, args.config)

    if isinstance(args.type_list, list):
        args.type_list = [str(x).strip() for x in args.type_list if str(x).strip()]
    else:
        args.type_list = [x.strip() for x in args.type_list.split(",") if x.strip()]
    if isinstance(args.site_topk, list):
        args.site_topk = [int(x) for x in args.site_topk]
    else:
        args.site_topk = parse_topk(args.site_topk)

    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)
        if args.log_dir:
            args.log_dir = os.path.join(args.log_dir, args.run_name)

    if args.build_mmap:
        if args.mmap_dir is None:
            raise ValueError("mmap_dir must be provided to build mmap.")
        build_router_mmap(
            args.train_jsonl,
            args.pdb_dir,
            args.mmap_dir,
            args.type_list,
            chain_id=args.chain_id,
            backbone_only=args.backbone_only,
        )
        print_log("mmap dataset created, exiting.")
        return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RouterDataset(
        jsonl_path=args.train_jsonl,
        pdb_dir=args.pdb_dir,
        type_list=args.type_list,
        chain_id=args.chain_id,
        backbone_only=args.backbone_only,
        use_mmap=args.use_mmap,
        mmap_dir=args.mmap_dir,
    )
    if args.train_ids or args.valid_ids:
        if not (args.train_ids and args.valid_ids):
            raise ValueError("Both --train_ids and --valid_ids must be provided.")
        train_ids = set(load_id_list(args.train_ids))
        valid_ids = set(load_id_list(args.valid_ids))
        overlap = train_ids.intersection(valid_ids)
        if overlap:
            print_log(f"Warning: {len(overlap)} ids appear in both train and valid; dropping from train.")
            train_ids = train_ids - overlap
        id_to_idx = dataset.build_id_index()
        train_idx = [id_to_idx[_id] for _id in train_ids if _id in id_to_idx]
        val_idx = [id_to_idx[_id] for _id in valid_ids if _id in id_to_idx]
        missing_train = len(train_ids) - len(train_idx)
        missing_valid = len(valid_ids) - len(val_idx)
        if missing_train:
            print_log(f"Warning: {missing_train} train ids not found in dataset.")
        if missing_valid:
            print_log(f"Warning: {missing_valid} valid ids not found in dataset.")
    else:
        train_idx, val_idx = split_indices(len(dataset), args.val_ratio, args.seed)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    if args.search:
        lr_list = parse_list(args.search_lr, float)
        wd_list = parse_list(args.search_weight_decay, float)
        bs_list = parse_list(args.search_batch_size, int)
        if not lr_list:
            lr_list = [args.learning_rate]
        if not wd_list:
            wd_list = [args.weight_decay]
        if not bs_list:
            bs_list = [args.batch_size]
        best = {"metric": -1e9, "config": None}
        search_results = []
        trial = 0
        for lr in lr_list:
            for wd in wd_list:
                for bs in bs_list:
                    trial += 1
                    args.learning_rate = lr
                    args.weight_decay = wd
                    args.batch_size = bs
                    args.output_dir = os.path.join(args.output_dir, f"trial_{trial}")
                    args.log_dir = os.path.join(args.log_dir, f"trial_{trial}")
                    print_log(f"Search trial {trial}: lr={lr} wd={wd} bs={bs}")
                    results = run_training(args, train_set, val_set, device, args.max_peptide_len)
                    trial_payload = {
                        "trial": trial,
                        "learning_rate": lr,
                        "weight_decay": wd,
                        "batch_size": bs,
                        "results": results,
                    }
                    search_results.append(trial_payload)
                    trial_name = args.result_name or "search"
                    trial_path = os.path.join(args.result_dir, f"{trial_name}_trial_{trial}.json")
                    save_json(trial_path, trial_payload)
                    metric = results.get("joint", {}).get("best_metric", None)
                    if metric is None:
                        metric = results.get("stage2", {}).get("best_metric", -1e9)
                    if metric > best["metric"]:
                        best = {"metric": metric, "config": {"lr": lr, "wd": wd, "bs": bs}}
        summary_name = args.result_name or "search_summary"
        summary_path = os.path.join(args.result_dir, f"{summary_name}.json")
        save_json(summary_path, {"best": best, "trials": search_results})
        print_log(f"Search best metric={best['metric']:.4f} config={best['config']}")
        return

    results = run_training(args, train_set, val_set, device, args.max_peptide_len)
    result_name = args.result_name or args.run_name or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    result_path = os.path.join(args.result_dir, f"{result_name}.json")
    payload = {"args": vars(args), "results": results}
    save_json(result_path, payload)


if __name__ == "__main__":
    main()
