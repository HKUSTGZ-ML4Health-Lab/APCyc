#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from utils.logger import print_log
from apcyc.evaluation.io import pdb_utils


def _which(cmd: Optional[str]) -> Optional[str]:
    if not cmd:
        return None
    if os.path.isabs(cmd) and os.path.exists(cmd):
        return cmd
    return shutil.which(cmd)


def _write_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:  # pylint: disable=broad-except
        return None


def _save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _chain_residue_range(pdb_path: str, chain_id: str) -> Optional[Tuple[int, int]]:
    try:
        structure = pdb_utils.load_structure(pdb_path)
        chain = pdb_utils.get_chain(structure, chain_id)
        if chain is None:
            return None
        residues = pdb_utils.residues_as_list(chain)
        if not residues:
            return None
        res_ids = [res.id[1] for res in residues if isinstance(res.id[1], int)]
        if not res_ids:
            return None
        return min(res_ids), max(res_ids)
    except Exception:  # pylint: disable=broad-except
        return None


def _build_tleap_input(pdb_path: str, cfg: dict, out_prefix: str) -> str:
    ff = cfg.get('forcefield', 'ff14SB')
    water = cfg.get('water_model', 'tip3p')
    box_type = cfg.get('box_type', 'oct')
    buffer = cfg.get('buffer', 12.0)
    neutralize = cfg.get('neutralize', True)
    salt_pairs = cfg.get('salt_pairs', None)

    lines = [
        f"source leaprc.protein.{ff}",
        f"source leaprc.water.{water}",
        "source leaprc.gaff",
        "set default PBRadii mbondi3",
        f"complex = loadpdb {os.path.basename(pdb_path)}",
    ]
    if box_type == 'box':
        lines.append(f"solvateBox complex {water.upper()}BOX {buffer}")
    else:
        lines.append(f"solvateOct complex {water.upper()}BOX {buffer}")
    if neutralize:
        lines.append("addionsrand complex Na+ 0")
        lines.append("addionsrand complex Cl- 0")
    if salt_pairs is not None:
        lines.append(f"addionsrand complex Na+ {int(salt_pairs)}")
        lines.append(f"addionsrand complex Cl- {int(salt_pairs)}")
    lines.extend([
        f"saveamberparm complex {out_prefix}.prmtop {out_prefix}.inpcrd",
        f"savepdb complex {out_prefix}.leap.pdb",
        "quit",
    ])
    return "\n".join(lines) + "\n"


def _min_input(stage: str, cfg: dict, restrained: bool) -> str:
    maxcyc = int(cfg.get('maxcyc', 4000))
    ncyc = int(cfg.get('ncyc', maxcyc // 2))
    cut = float(cfg.get('cutoff', 9.0))
    restraint_wt = float(cfg.get('restraint_wt', 8.0))
    restraint_mask = cfg.get('restraint_mask', '!:WAT & !:Na+ & !:Cl-')

    lines = [
        f"{stage}",
        "&cntrl",
        "  imin=1,",
        f"  maxcyc={maxcyc}, ncyc={ncyc},",
        f"  cut={cut:.2f},",
        "  ntb=1,",
        "  ntpr=100,",
    ]
    if restrained:
        lines.append("  ntr=1,")
        lines.append(f"  restraint_wt={restraint_wt:.2f},")
        lines.append(f"  restraintmask='{restraint_mask}',")
    lines.append("/\n")
    return "\n".join(lines)


def _heat_input(cfg: dict) -> str:
    dt = float(cfg.get('dt', 0.002))
    time_ps = float(cfg.get('time_ps', 400.0))
    nstlim = int(round((time_ps / 1000.0) / dt))
    t0 = float(cfg.get('t0', 0.0))
    t1 = float(cfg.get('t1', 310.0))
    cut = float(cfg.get('cutoff', 9.0))
    restraint_wt = float(cfg.get('restraint_wt', 8.0))
    restraint_mask = cfg.get('restraint_mask', '!:WAT & !:Na+ & !:Cl-')
    gamma_ln = float(cfg.get('gamma_ln', 1.0))

    lines = [
        "Heat",
        "&cntrl",
        f"  nstlim={nstlim}, dt={dt:.3f},",
        "  imin=0, irest=0, ntx=1,",
        "  ntb=1, ntp=0,",
        f"  tempi={t0:.1f}, temp0={t1:.1f},",
        f"  cut={cut:.2f},",
        "  ntc=2, ntf=2,",
        "  ntt=3,",
        f"  gamma_ln={gamma_ln:.1f},",
        "  ntpr=500, ntwx=500,",
        "  ntr=1,",
        f"  restraint_wt={restraint_wt:.2f},",
        f"  restraintmask='{restraint_mask}',",
        "/\n",
    ]
    return "\n".join(lines)


def _equil_input(stage_idx: int, cfg: dict, restraint_wt: float) -> str:
    dt = float(cfg.get('dt', 0.002))
    time_ps = float(cfg.get('stage_time_ps', 400.0))
    nstlim = int(round((time_ps / 1000.0) / dt))
    temp = float(cfg.get('temp', 300.0))
    pressure = float(cfg.get('pressure', 1.0))
    cut = float(cfg.get('cutoff', 9.0))
    gamma_ln = float(cfg.get('gamma_ln', 1.0))
    barostat = int(cfg.get('barostat', 2))
    restraint_mask = cfg.get('restraint_mask', '!:WAT & !:Na+ & !:Cl-')

    lines = [
        f"Equil {stage_idx}",
        "&cntrl",
        f"  nstlim={nstlim}, dt={dt:.3f},",
        "  imin=0, irest=1, ntx=5,",
        "  ntb=2, ntp=1,",
        f"  temp0={temp:.1f}, pres0={pressure:.1f},",
        f"  cut={cut:.2f},",
        "  ntc=2, ntf=2,",
        "  ntt=3,",
        f"  gamma_ln={gamma_ln:.1f},",
        "  ntpr=500, ntwx=500,",
        "  ntr=1,",
        f"  restraint_wt={restraint_wt:.2f},",
        f"  restraintmask='{restraint_mask}',",
        f"  barostat={barostat},",
        "/\n",
    ]
    return "\n".join(lines)


def _prod_input(cfg: dict) -> str:
    dt = float(cfg.get('dt', 0.002))
    time_ns = float(cfg.get('time_ns', 30.0))
    nstlim = int(round((time_ns) / dt * 1000.0))
    temp = float(cfg.get('temp', 300.0))
    pressure = float(cfg.get('pressure', 1.0))
    cut = float(cfg.get('cutoff', 9.0))
    gamma_ln = float(cfg.get('gamma_ln', 1.0))
    barostat = int(cfg.get('barostat', 2))
    ntpr = int(cfg.get('ntpr', 1000))
    ntwx = int(cfg.get('ntwx', 1000))

    lines = [
        "Production",
        "&cntrl",
        f"  nstlim={nstlim}, dt={dt:.3f},",
        "  imin=0, irest=1, ntx=5,",
        "  ntb=2, ntp=1,",
        f"  temp0={temp:.1f}, pres0={pressure:.1f},",
        f"  cut={cut:.2f},",
        "  ntc=2, ntf=2,",
        "  ntt=3,",
        f"  gamma_ln={gamma_ln:.1f},",
        f"  ntpr={ntpr}, ntwx={ntwx},",
        f"  barostat={barostat},",
        "/\n",
    ]
    return "\n".join(lines)


def _rmsd_cpptraj_input(mask: str, out_path: str) -> str:
    return "\n".join([
        "trajin prod.nc",
        "reference prod.nc 1",
        f"rms reference {mask} out {os.path.basename(out_path)}",
        "run",
        "quit",
        "",
    ])


def _parse_rmsd(path: str) -> Tuple[Optional[float], Optional[float]]:
    if not os.path.exists(path):
        return None, None
    vals = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            try:
                vals.append(float(parts[1]))
            except ValueError:
                continue
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return float(mean), float(var ** 0.5)


def _parse_mmpbsa(path: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        for line in f:
            if line.strip().startswith("DELTA TOTAL"):
                parts = line.split()
                try:
                    return float(parts[-1])
                except Exception:  # pylint: disable=broad-except
                    continue
    return None


def _run(cmd: List[str], timeout: Optional[int], log_path: Optional[str], cwd: Optional[str] = None) -> Tuple[bool, str]:
    try:
        with open(log_path, 'w') if log_path else open(os.devnull, 'w') as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=logf, timeout=timeout, cwd=cwd)
        return True, 'ok'
    except subprocess.TimeoutExpired:
        return False, 'timeout'
    except subprocess.CalledProcessError:
        return False, 'failed'
    except Exception as exc:  # pylint: disable=broad-except
        print_log(f"MD command failed: {exc}")
        return False, 'failed'


def _normalize_cmd_prefix(prefix) -> List[str]:
    if not prefix:
        return []
    if isinstance(prefix, str):
        return prefix.split()
    if isinstance(prefix, (list, tuple)):
        return [str(p) for p in prefix if str(p).strip()]
    return []


def build_run_dir(base_dir: str, target_id: str, sample_id: str) -> str:
    safe_target = str(target_id).replace(os.sep, '_')
    safe_sample = str(sample_id).replace(os.sep, '_')
    return os.path.join(base_dir, safe_target, safe_sample)


def should_run_md(target_id: str, sample: dict, cfg: dict, md_state: dict) -> bool:
    if not cfg.get('enabled', False):
        return False
    target_ids = cfg.get('target_ids')
    if target_ids and str(target_id) not in {str(t) for t in target_ids}:
        return False
    types = cfg.get('types')
    if types is not None:
        type_hat = sample.get('type_hat')
        if type_hat not in types:
            return False
    sample_ids = cfg.get('sample_ids')
    if sample_ids and str(sample.get('sample_id')) not in {str(s) for s in sample_ids}:
        return False
    max_per_target = cfg.get('max_samples_per_target', 1)
    try:
        max_per_target = int(max_per_target)
    except (TypeError, ValueError):
        max_per_target = 1
    if max_per_target > 0 and md_state.get('count', 0) >= max_per_target:
        return False
    return True


def run_md(
    pdb_path: str,
    pep_chain: str,
    rec_chain: str,
    cfg: dict,
    out_dir: str,
) -> Tuple[Dict[str, Optional[float]], str]:
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'md_results.json')
    cached = _load_json(result_path)
    if cached and cached.get('status') == 'ok':
        return cached.get('metrics', {}), 'cached'

    run_mode = cfg.get('run_mode', 'write')
    amber_cfg = cfg.get('amber', {})
    tleap_bin = _which(amber_cfg.get('tleap') or os.environ.get('AMBER_TLEAP') or 'tleap')
    pmemd_bin = _which(amber_cfg.get('pmemd') or os.environ.get('AMBER_PMEMD') or 'pmemd.cuda')
    cpptraj_bin = _which(amber_cfg.get('cpptraj') or os.environ.get('AMBER_CPPTRAJ') or 'cpptraj')
    mmpbsa_bin = _which(amber_cfg.get('mmpbsa') or os.environ.get('AMBER_MMPBSA') or 'MMPBSA.py')
    ante_mmpbsa_bin = _which(amber_cfg.get('ante_mmpbsa') or os.environ.get('AMBER_ANTE_MMPBSA') or 'ante-MMPBSA.py')
    timeout = cfg.get('timeout')
    pmemd_prefix = _normalize_cmd_prefix(amber_cfg.get('pmemd_prefix') or amber_cfg.get('mpi_cmd'))

    work_pdb = os.path.join(out_dir, os.path.basename(pdb_path))
    if not os.path.exists(work_pdb):
        shutil.copy(pdb_path, work_pdb)

    system_cfg = cfg.get('system', {})
    min_cfg = cfg.get('minimization', {})
    heat_cfg = cfg.get('heating', {})
    equil_cfg = cfg.get('equilibration', {})
    prod_cfg = cfg.get('production', {})
    analysis_cfg = cfg.get('analysis', {})

    complex_prefix = os.path.join(out_dir, 'complex')
    tleap_in = os.path.join(out_dir, 'tleap.in')
    _write_text(tleap_in, _build_tleap_input(work_pdb, system_cfg, os.path.basename(complex_prefix)))

    min1_in = os.path.join(out_dir, 'min1.in')
    min2_in = os.path.join(out_dir, 'min2.in')
    heat_in = os.path.join(out_dir, 'heat.in')
    _write_text(min1_in, _min_input("Minimize solvent/ions", min_cfg, restrained=True))
    _write_text(min2_in, _min_input("Minimize all", min_cfg, restrained=False))
    _write_text(heat_in, _heat_input(heat_cfg))

    restraint_schedule = equil_cfg.get('restraint_schedule', [4.0, 2.0, 1.0, 0.5, 0.1])
    equil_inputs = []
    for idx, wt in enumerate(restraint_schedule, start=1):
        eq_in = os.path.join(out_dir, f'equil_{idx}.in')
        _write_text(eq_in, _equil_input(idx, equil_cfg, float(wt)))
        equil_inputs.append(eq_in)

    prod_in = os.path.join(out_dir, 'prod.in')
    _write_text(prod_in, _prod_input(prod_cfg))

    rmsd_cfg = analysis_cfg.get('rmsd', {})
    rmsd_out = os.path.join(out_dir, 'rmsd.dat')
    if rmsd_cfg.get('enabled', True):
        mask = rmsd_cfg.get('mask')
        if not mask:
            res_range = _chain_residue_range(work_pdb, pep_chain)
            if res_range:
                mask = f":{res_range[0]}-{res_range[1]}@CA"
        if mask:
            cpptraj_in = os.path.join(out_dir, 'cpptraj.in')
            _write_text(cpptraj_in, _rmsd_cpptraj_input(mask, rmsd_out))

    mmpbsa_cfg = analysis_cfg.get('mmpbsa', {})
    mmpbsa_in = os.path.join(out_dir, 'mmpbsa.in')
    if mmpbsa_cfg.get('enabled', False):
        _write_text(mmpbsa_in, mmpbsa_cfg.get('input', "MMPBSA\n&general\n  endframe=-1,\n/\n"))

    run_sh = os.path.join(out_dir, 'run_md.sh')
    pmemd_exec = " ".join(pmemd_prefix + [pmemd_bin or 'pmemd.cuda'])
    run_lines = [
        "#!/usr/bin/env bash",
        "set -e",
        f"cd {out_dir}",
        "",
        f"{tleap_bin or 'tleap'} -f {os.path.basename(tleap_in)}",
        "",
        f"{pmemd_exec} -O -i {os.path.basename(min1_in)} -o min1.out -p complex.prmtop "
        f"-c complex.inpcrd -r min1.rst7 -ref complex.inpcrd",
        f"{pmemd_exec} -O -i {os.path.basename(min2_in)} -o min2.out -p complex.prmtop "
        f"-c min1.rst7 -r min2.rst7",
        f"{pmemd_exec} -O -i {os.path.basename(heat_in)} -o heat.out -p complex.prmtop "
        f"-c min2.rst7 -r heat.rst7 -ref min2.rst7",
    ]
    prev = "heat.rst7"
    for eq_in in equil_inputs:
        eq_base = os.path.splitext(os.path.basename(eq_in))[0]
        run_lines.append(
            f"{pmemd_exec} -O -i {os.path.basename(eq_in)} -o {eq_base}.out -p complex.prmtop "
            f"-c {prev} -r {eq_base}.rst7 -ref {prev}"
        )
        prev = f"{eq_base}.rst7"
    run_lines.append(
        f"{pmemd_exec} -O -i {os.path.basename(prod_in)} -o prod.out -p complex.prmtop "
        f"-c {prev} -r prod.rst7 -x prod.nc"
    )
    if rmsd_cfg.get('enabled', True):
        run_lines.append(f"{cpptraj_bin or 'cpptraj'} -p complex.prmtop -i cpptraj.in")
    if mmpbsa_cfg.get('enabled', False):
        if mmpbsa_cfg.get('ante_cmd'):
            run_lines.append(" ".join(mmpbsa_cfg.get('ante_cmd')))
        elif mmpbsa_cfg.get('auto_ante', False) and ante_mmpbsa_bin:
            rec_mask = mmpbsa_cfg.get('receptor_mask')
            lig_mask = mmpbsa_cfg.get('ligand_mask')
            if rec_mask and lig_mask:
                run_lines.append(
                    f"{ante_mmpbsa_bin} -p complex.prmtop -r receptor.prmtop -l ligand.prmtop "
                    f"-c complex.prmtop -s '{rec_mask}' -m '{lig_mask}'"
                )
        if mmpbsa_cfg.get('command'):
            run_lines.append(" ".join(mmpbsa_cfg.get('command')))
        elif mmpbsa_bin:
            run_lines.append(
                f"{mmpbsa_bin} -O -i {os.path.basename(mmpbsa_in)} -cp complex.prmtop "
                f"-rp receptor.prmtop -lp ligand.prmtop -y prod.nc"
            )
    _write_text(run_sh, "\n".join(run_lines) + "\n")
    os.chmod(run_sh, 0o755)

    metrics = {
        'md_rmsd_mean': None,
        'md_rmsd_std': None,
        'md_mmpbsa_dg': None,
    }

    if run_mode == 'write':
        result = {'status': 'written', 'metrics': metrics, 'out_dir': out_dir}
        _save_json(result_path, result)
        return metrics, 'written'

    if not tleap_bin or not pmemd_bin:
        return metrics, 'missing_amber'

    ok, status = _run([tleap_bin, '-f', os.path.basename(tleap_in)], timeout, os.path.join(out_dir, 'tleap.log'), cwd=out_dir)
    if not ok:
        return metrics, f'tleap_{status}'

    if run_mode == 'prepare':
        result = {'status': 'prepared', 'metrics': metrics, 'out_dir': out_dir}
        _save_json(result_path, result)
        return metrics, 'prepared'

    def _md_step(cmd, log_name):
        ok_step, st = _run(cmd, timeout, os.path.join(out_dir, log_name), cwd=out_dir)
        return ok_step, st

    ok, st = _md_step(pmemd_prefix + [pmemd_bin, '-O', '-i', os.path.basename(min1_in), '-o', 'min1.out',
                       '-p', 'complex.prmtop', '-c', 'complex.inpcrd',
                       '-r', 'min1.rst7', '-ref', 'complex.inpcrd'],
                      'min1.log')
    if not ok:
        return metrics, f'min1_{st}'
    ok, st = _md_step(pmemd_prefix + [pmemd_bin, '-O', '-i', os.path.basename(min2_in), '-o', 'min2.out',
                       '-p', 'complex.prmtop', '-c', 'min1.rst7',
                       '-r', 'min2.rst7'], 'min2.log')
    if not ok:
        return metrics, f'min2_{st}'
    ok, st = _md_step(pmemd_prefix + [pmemd_bin, '-O', '-i', os.path.basename(heat_in), '-o', 'heat.out',
                       '-p', 'complex.prmtop', '-c', 'min2.rst7',
                       '-r', 'heat.rst7', '-ref', 'min2.rst7'],
                      'heat.log')
    if not ok:
        return metrics, f'heat_{st}'

    prev_rst = 'heat.rst7'
    for eq_in in equil_inputs:
        eq_base = os.path.splitext(os.path.basename(eq_in))[0]
        eq_out = f'{eq_base}.out'
        eq_rst = f'{eq_base}.rst7'
        ok, st = _md_step(pmemd_prefix + [pmemd_bin, '-O', '-i', os.path.basename(eq_in), '-o', eq_out,
                           '-p', 'complex.prmtop', '-c', prev_rst,
                           '-r', eq_rst, '-ref', prev_rst], f'{eq_base}.log')
        if not ok:
            return metrics, f'{eq_base}_{st}'
        prev_rst = eq_rst

    ok, st = _md_step(pmemd_prefix + [pmemd_bin, '-O', '-i', os.path.basename(prod_in), '-o', 'prod.out',
                       '-p', 'complex.prmtop', '-c', prev_rst,
                       '-r', 'prod.rst7', '-x', 'prod.nc'],
                      'prod.log')
    if not ok:
        return metrics, f'prod_{st}'

    if rmsd_cfg.get('enabled', True) and cpptraj_bin and os.path.exists(os.path.join(out_dir, 'cpptraj.in')):
        ok, st = _run([cpptraj_bin, '-p', 'complex.prmtop',
                       '-i', 'cpptraj.in'], timeout, os.path.join(out_dir, 'cpptraj.log'), cwd=out_dir)
        if ok:
            metrics['md_rmsd_mean'], metrics['md_rmsd_std'] = _parse_rmsd(rmsd_out)

    if mmpbsa_cfg.get('enabled', False) and mmpbsa_bin:
        mmpbsa_cmd = mmpbsa_cfg.get('command')
        if mmpbsa_cmd:
            cmd = [part.format(
                mmpbsa_in=mmpbsa_in,
                complex_prmtop=os.path.join(out_dir, 'complex.prmtop'),
                receptor_prmtop=os.path.join(out_dir, 'receptor.prmtop'),
                ligand_prmtop=os.path.join(out_dir, 'ligand.prmtop'),
                prod_nc=os.path.join(out_dir, 'prod.nc'),
            ) for part in mmpbsa_cmd]
            _run(cmd, timeout, os.path.join(out_dir, 'mmpbsa.log'), cwd=out_dir)
        else:
            _run([mmpbsa_bin, '-O', '-i', os.path.basename(mmpbsa_in),
                  '-cp', 'complex.prmtop',
                  '-rp', 'receptor.prmtop',
                  '-lp', 'ligand.prmtop',
                  '-y', 'prod.nc'],
                 timeout, os.path.join(out_dir, 'mmpbsa.log'), cwd=out_dir)
        metrics['md_mmpbsa_dg'] = _parse_mmpbsa(os.path.join(out_dir, 'FINAL_RESULTS_MMPBSA.dat'))

    result = {'status': 'ok', 'metrics': metrics, 'out_dir': out_dir}
    _save_json(result_path, result)
    return metrics, 'ok'
