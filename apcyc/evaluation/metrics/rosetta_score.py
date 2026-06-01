#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import subprocess
import tempfile
from typing import Optional, Sequence, Tuple, Union

from utils.logger import print_log


def _normalize_chains(chain: Optional[Union[Sequence[str], str]]) -> list:
    if chain is None:
        return []
    if isinstance(chain, str):
        return [chain]
    return [str(c) for c in chain]


def _extract_chain_pose(pose, chain_ids: Sequence[str]):
    if not chain_ids:
        return None
    pdb_info = pose.pdb_info()
    chain_indices = []
    for chain_idx in range(1, pose.num_chains() + 1):
        start = pose.chain_begin(chain_idx)
        if pdb_info.chain(start) in chain_ids:
            chain_indices.append(chain_idx)
    if not chain_indices:
        return None
    split = pose.split_by_chain()
    new_pose = None
    for chain_idx in chain_indices:
        subpose = split[chain_idx]
        if new_pose is None:
            new_pose = subpose.clone()
        else:
            from pyrosetta.rosetta.core.pose import append_pose_to_pose
            append_pose_to_pose(new_pose, subpose, True)
    return new_pose


def _extract_score_from_scorefile(path: str, key: str) -> Optional[float]:
    key_re = re.compile(rf'"{re.escape(key)}"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                match = key_re.search(line)
                if match:
                    return float(match.group(1))
    except Exception:  # pylint: disable=broad-except
        return None
    return None


def _extract_dg_from_scorefile(path: str) -> Optional[float]:
    return _extract_score_from_scorefile(path, 'dg')


def _rosetta_scripts_score(
    pdb_path: str,
    rosetta_bin: str,
    rosetta_xml: str,
    timeout: Optional[int],
    key: str,
) -> Tuple[Optional[float], str]:
    if not os.path.exists(rosetta_bin):
        return None, 'missing_rosetta_bin'
    if not os.path.exists(rosetta_xml):
        return None, 'missing_rosetta_xml'
    with tempfile.TemporaryDirectory() as tmpdir:
        scorefile = os.path.join(tmpdir, 'rosetta.sc')
        cmd = [
            rosetta_bin,
            "-s",
            pdb_path,
            "-in:file:native",
            pdb_path,
            "-parser:protocol",
            rosetta_xml,
            "-out:file:scorefile",
            scorefile,
            "-out:file:scorefile_format",
            "json",
            "-out:file:score_only",
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, 'rosetta_timeout'
        except subprocess.CalledProcessError:
            return None, 'rosetta_failed'
        score = _extract_score_from_scorefile(scorefile, key)
        if score is None:
            return None, f'rosetta_no_{key}'
        return score, 'ok'


def _rosetta_scripts_interface_dg(
    pdb_path: str,
    rosetta_bin: str,
    rosetta_xml: str,
    timeout: Optional[int],
) -> Tuple[Optional[float], str]:
    return _rosetta_scripts_score(pdb_path, rosetta_bin, rosetta_xml, timeout, 'dg')


def _rosetta_scripts_total_score(
    pdb_path: str,
    rosetta_bin: str,
    rosetta_xml: str,
    timeout: Optional[int],
) -> Tuple[Optional[float], str]:
    return _rosetta_scripts_score(pdb_path, rosetta_bin, rosetta_xml, timeout, 'total_score')


def rosetta_total_score(
    pdb_path: str,
    pep_chain: str,
    config: dict,
    rec_chain: Optional[Union[Sequence[str], str]] = None,
) -> Tuple[Optional[float], str]:
    if not config.get('enabled', True):
        return None, 'disabled'
    relax = config.get('relax', False)
    energy_mode = config.get('energy_mode', 'interface_dg')
    engine = config.get('engine', 'pyrosetta')
    scorefxn_name = config.get('scorefxn', 'ref2015')
    total_score_scope = config.get('total_score_scope', 'peptide')
    timeout = config.get('timeout', None)

    try:
        if energy_mode == 'total_score' and total_score_scope == 'peptide' and engine == 'rosetta_scripts':
            # Peptide-only total score requires PyRosetta to score the peptide subpose.
            engine = 'pyrosetta'
        if energy_mode == 'interface_dg' and engine == 'rosetta_scripts':
            rosetta_bin = config.get('rosetta_bin') or os.environ.get('ROSETTA_BIN')
            rosetta_xml = config.get('rosetta_xml') or os.environ.get('ROSETTA_XML')
            if not rosetta_bin or not rosetta_xml:
                return None, 'missing_rosetta_bin'
            return _rosetta_scripts_interface_dg(pdb_path, rosetta_bin, rosetta_xml, timeout)
        if energy_mode == 'total_score' and engine == 'rosetta_scripts':
            rosetta_bin = config.get('rosetta_bin') or os.environ.get('ROSETTA_BIN')
            rosetta_xml = config.get('rosetta_xml') or os.environ.get('ROSETTA_XML')
            if not rosetta_bin or not rosetta_xml:
                return None, 'missing_rosetta_bin'
            return _rosetta_scripts_total_score(pdb_path, rosetta_bin, rosetta_xml, timeout)

        try:
            from evaluation.dG import energy
        except Exception as exc:  # pylint: disable=broad-except
            print_log(f"Rosetta unavailable: {exc}")
            return None, 'missing_rosetta'
        if relax:
            if energy_mode == 'interface_dg':
                pep_chains = _normalize_chains(pep_chain)
                rec_chains = _normalize_chains(rec_chain)
                if not rec_chains:
                    return None, 'missing_receptor'
                rfdiff_relax = bool(config.get('rfdiff_relax', False))
                with tempfile.TemporaryDirectory() as tmpdir:
                    relaxed_path = os.path.join(tmpdir, "relaxed.pdb")
                    if rfdiff_relax:
                        energy.rfdiff_refine(pdb_path, relaxed_path, pep_chain)
                    else:
                        energy.pyrosetta_fastrelax(pdb_path, relaxed_path, pep_chain, rfdiff_config=rfdiff_relax)
                    score = float(energy.pyrosetta_interface_energy(relaxed_path, rec_chains, pep_chains))
            else:
                subset = config.get('subset', 'nbrs')
                move_bb = config.get('move_bb', False)
                max_iter = config.get('max_iter', 1000)
                relaxer = energy.RelaxRegion(scorefxn=scorefxn_name, max_iter=max_iter, subset=subset, move_bb=move_bb)
                pose, _, e_relax = relaxer(pdb_path, ligand_chains=[pep_chain])
                score = float(e_relax)
        else:
            if energy_mode == 'total_score':
                import pyrosetta
                pose = pyrosetta.pose_from_pdb(pdb_path)
                scorefxn = energy.get_scorefxn(scorefxn_name)
                if total_score_scope == 'peptide':
                    pep_chains = _normalize_chains(pep_chain)
                    subpose = _extract_chain_pose(pose, pep_chains)
                    if subpose is None:
                        return None, 'missing_peptide'
                    score = float(scorefxn(subpose))
                else:
                    score = float(scorefxn(pose))
            else:
                pep_chains = _normalize_chains(pep_chain)
                rec_chains = _normalize_chains(rec_chain)
                if not rec_chains:
                    return None, 'missing_receptor'
                score = float(energy.pyrosetta_interface_energy(pdb_path, rec_chains, pep_chains))
        return score, 'ok'
    except Exception as exc:  # pylint: disable=broad-except
        print_log(f"Rosetta failed for {os.path.basename(pdb_path)}: {exc}")
        return None, 'rosetta_failed'
