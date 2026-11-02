"""Collect property measurements with isolated outputs, resumable stages and missingness records."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser, Select
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import seq1
import yaml

from pipelines.process_utils import run_isolated
from utils.file_utils import sha256


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def load_inputs(manifest):
    records, seen = [], set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pdb = Path(r.get('pdb', r.get('complex_pdb', '')))
        pdb = pdb if pdb.is_absolute() else manifest.parent / pdb
        sid = r['id'] if 'id' in r else pdb.stem
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', sid) or sid in seen:
            raise ValueError(f'Invalid or duplicate property sample ID: {sid}')
        seen.add(sid)
        records.append(dict(id=sid, pdb=pdb.resolve(), chain=r.get('peptide_chain', 'L'),
                            receptor_id=r.get('receptor_id', r.get('sample_id', sid)), sha256=sha256(pdb)))
        if 'sequence' in r:
            records[-1]['expected_sequence'] = r['sequence']
    if not records:
        raise ValueError('Empty structure manifest')
    return records


def prepare(records, out):
    if any(r['pdb'].resolve().is_relative_to(out.resolve()) for r in records):
        raise ValueError('Property output directory must be separate from source structures.')
    structures = out / 'inputs'
    zips = out / 'structure_zips'
    peptides = out / 'peptides'
    for p in [structures, zips, peptides]:
        p.mkdir(parents=True, exist_ok=True)
        if not p.resolve().is_relative_to(out.resolve()):
            raise ValueError('Prepared structures must remain inside the output directory.')
    rows, failures = [], []
    fasta, pdb_list = [], []
    for r in records:
        target = structures / (r['id'] + '.pdb')
        pep_path, zip_path = peptides / (r['id'] + '.pdb'), zips / (r['id'] + '.zip')
        row = dict(id=r['id'], sequence='', GRAVY='')
        try:
            structure = PDBParser(QUIET=True).get_structure(r['id'], str(r['pdb']))
            if len(structure) != 1:
                raise ValueError('Property collection requires exactly one PDB model.')
            chain = structure[0][r['chain']]
            residues = [a for a in chain if a.id[0] == ' ' or a.resname == 'MSE']
            if any(not all(atom in a for atom in ['N', 'CA', 'C', 'O']) for a in residues):
                raise ValueError('Incomplete peptide backbone.')
            sequence = ''.join(seq1(a.resname, custom_map={'MSE': 'M'}) for a in residues)
            if not sequence or any(a not in 'ACDEFGHIKLMNPQRSTVWY' for a in sequence):
                raise ValueError('Peptide sequence is empty or contains unsupported residues')
            if 'expected_sequence' in r and r['expected_sequence'] != sequence:
                raise ValueError('Manifest sequence differs from the property PDB.')
            if not all(math.isfinite(float(x)) for atom in structure.get_atoms() for x in atom.coord):
                raise ValueError('Nonfinite structure coordinates.')
            class PeptideOnly(Select):
                def accept_chain(self, c):
                    return c.id == r['chain']
                def accept_residue(self, residue):
                    return residue in residues
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(pep_path), PeptideOnly())
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
                z.write(pep_path, arcname=pep_path.name)
            shutil.copyfile(r['pdb'], target)
            row.update(sequence=sequence, GRAVY=ProteinAnalysis(sequence).gravy())
            fasta.append(f'>{r["id"]}\n{sequence}\n')
            pdb_list.append(str(target))
        except Exception as exc:
            for path in [target, pep_path, zip_path]:
                if path.is_file():
                    path.unlink()
            failures.append(dict(id=r['id'], stage='prepare', error=f'{type(exc).__name__}: {exc}'))
        rows.append(row)
    (out / 'sequences.fasta').write_text(''.join(fasta))
    (out / 'pdb_list.txt').write_text('\n'.join(pdb_list) + '\n')
    return rows, failures


def expand(value, context):
    rendered = os.path.expandvars(str(value))
    if re.search(r'\$\{?\w+', rendered):
        raise ValueError('Unset environment variable in stage configuration')
    rendered = rendered.format_map(context)
    if re.search(r'\$\{?\w+', rendered):
        raise ValueError('Unset environment variable in stage configuration')
    return rendered


def merge_stage(path, spec, ids):
    result = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=spec.get('delimiter', ','))
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('Missing or duplicate stage output headers.')
        required = {spec.get('id_column', 'id')} | set(spec['columns'].values())
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'Missing output columns: {sorted(required - set(reader.fieldnames or []))}')
        for row in reader:
            sid = row[spec.get('id_column', 'id')]
            if spec.get('id_normalization') == 'pdb_stem':
                sid = Path(sid).stem
            if sid not in ids:
                raise ValueError(f'Unexpected stage output ID: {sid}')
            if sid in result:
                raise ValueError(f'Duplicate stage output ID: {sid}')
            values = {}
            for key, column in spec['columns'].items():
                raw = row[column]
                try:
                    values[key] = raw if math.isfinite(float(raw)) else ''
                except (TypeError, ValueError):
                    values[key] = ''
            result[sid] = values
    return result


def collect(args):
    config = yaml.safe_load(args.config.read_text()) or {}
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = load_inputs(args.manifest.resolve())
    ids = {r['id'] for r in records}
    identity = [{k: str(v) for k, v in r.items() if k != 'expected_sequence'} for r in records]
    fingerprint = hashlib.sha256(json.dumps(dict(inputs=identity, config=config), sort_keys=True).encode()).hexdigest()
    prior = out / 'inputs.json'
    if prior.exists() and json.loads(prior.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Output directory belongs to different inputs/config; choose a new directory.')
    dump(prior, dict(fingerprint=fingerprint, records=identity))
    rows, failures = prepare(records, out)
    by_id = {r['id']: r for r in rows}
    valid_ids = {r['id'] for r in rows if r['sequence']}
    prepared_inputs = dict(fasta=sha256(out / 'sequences.fasta'),
        pdb_list=sha256(out / 'pdb_list.txt'),
        complexes={sid: sha256(out / 'inputs' / (sid + '.pdb')) for sid in sorted(valid_ids)},
        peptides={sid: sha256(out / 'peptides' / (sid + '.pdb')) for sid in sorted(valid_ids)})
    status = {'basic': dict(status='complete' if not failures else 'partial', rows=len(rows)-len(failures))}
    context = dict(output=str(out), inputs=str(out / 'inputs'), fasta=str(out / 'sequences.fasta'),
                   structures=str(out / 'structure_zips'), pdb_list=str(out / 'pdb_list.txt'),
                   repo=str(Path(__file__).resolve().parents[1]), chain=config.get('chain', 'L'))
    declared = set()
    stages = config.get('stages', {})
    if args.only and set(args.only) - ({'basic'} | set(stages)):
        raise ValueError('Unknown stage in --only')
    for name, spec in stages.items():
        columns = set(spec['columns'])
        if declared & columns or columns & {'id', 'sequence', 'GRAVY'}:
            raise ValueError(f'Overlapping output columns in stage {name}')
        declared |= columns
        if args.only and name not in args.only:
            status[name] = dict(status='skipped', reason='not selected')
            continue
        stage_dir = out / 'stages' / name
        stage_dir.mkdir(parents=True, exist_ok=True)
        ctx = dict(context, stage_dir=str(stage_dir), stage_output=str(stage_dir / 'results.csv'))
        try:
            output = Path(expand(spec.get('output', '{stage_output}'), ctx))
            if not output.resolve().is_relative_to(stage_dir.resolve()):
                raise ValueError('Stage outputs must remain inside their isolated stage directory.')
            output.parent.mkdir(parents=True, exist_ok=True)
            artifacts = {}
            if spec.get('import'):
                imported = Path(expand(spec['import'], ctx)).expanduser()
                imported = imported if imported.is_absolute() else args.config.resolve().parent / imported
                shutil.copyfile(imported, output)
                key = hashlib.sha256(json.dumps([fingerprint, sha256(imported), prepared_inputs], sort_keys=True).encode()).hexdigest()
                cache = False
            else:
                if not spec.get('enabled', False):
                    status[name] = dict(status='skipped', reason='disabled')
                    continue
                if not args.run_external:
                    status[name] = dict(status='unconfigured', reason='external execution not enabled')
                    continue
                artifact_paths = []
                for value in spec.get('artifacts', []):
                    path = Path(expand(value, ctx))
                    if not path.resolve().is_relative_to(stage_dir.resolve()):
                        raise ValueError('Stage artifacts must remain inside their isolated stage directory.')
                    artifact_paths.append(path)
                missing = [k for k in spec.get('required_env', []) if not os.environ.get(k)]
                if missing:
                    raise ValueError(f'Missing required environment names: {missing}')
                if len({r['chain'] for r in records}) != 1 or records[0]['chain'] != ctx['chain']:
                    raise ValueError('External stages require a uniform chain matching the configuration.')
                argv = [expand(a, ctx) for a in spec['argv']]
                if '--help' in argv or '-h' in argv:
                    raise ValueError('Help commands cannot be used as collection stages.')
                dependency_hashes = {}
                for value in spec.get('dependencies', []):
                    path = Path(expand(value, ctx))
                    dependency_hashes[str(path)] = sha256(path)
                key = hashlib.sha256(json.dumps([fingerprint, argv, dependency_hashes, prepared_inputs], sort_keys=True).encode()).hexdigest()
                marker = stage_dir / 'complete.json'
                previous = json.loads(marker.read_text()) if marker.exists() else {}
                cache = previous.get('status') == 'complete' and previous.get('key') == key and output.is_file() and previous.get('sha256') == sha256(output)
                cache = cache and all(path.is_file() and previous.get('artifacts', {}).get(str(path)) == sha256(path)
                                      for path in artifact_paths)
                if not cache:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if output.exists():
                        output.unlink()  # only files in this isolated stage directory are allowed below
                    for path in artifact_paths:
                        if path.exists():
                            path.unlink()
                    started = time.monotonic()
                    with (stage_dir / 'stdout.log').open('w') as stdout, (stage_dir / 'stderr.log').open('w') as stderr:
                        run_isolated(argv, cwd=stage_dir, stdout=stdout, stderr=stderr,
                                     timeout=spec.get('timeout_seconds', 3600))
                    if not output.is_file():
                        raise FileNotFoundError('Stage returned success without its expected output')
                for path in artifact_paths:
                    if not path.is_file():
                        raise FileNotFoundError(f'Stage returned success without its required artifact: {path.name}')
                    artifacts[str(path)] = sha256(path)
            # Imported tool output is retained; no missing results are imputed.
            result = merge_stage(output, spec, ids)
            for sid, values in result.items():
                if sid in valid_ids:
                    by_id[sid].update(values)
            complete_ids = {sid for sid, values in result.items() if sid in valid_ids and all(v != '' for v in values.values())}
            stage_report = dict(status='complete' if complete_ids == ids else 'partial', rows=len(result),
                                missing_ids=sorted(ids - complete_ids), sha256=sha256(output), artifacts=artifacts, cached=cache)
            status[name] = stage_report
            dump(stage_dir / 'complete.json', dict(key=key, sha256=sha256(output), artifacts=artifacts, status=stage_report['status']))
        except Exception as exc:
            status[name] = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
            failures.append(dict(stage=name, error=status[name]['error']))
    columns = ['id', 'sequence', 'GRAVY'] + sorted(declared)
    with (out / 'raw_properties.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    missingness = {col: sum(str(row.get(col, '')).strip().lower() in {'', 'nan', 'none', 'inf', '-inf'} for row in rows) for col in columns[2:]}
    dump(out / 'status.json', dict(stages=status, failures=failures, missing_counts=missingness,
                                 samples=len(rows), input_fingerprint=fingerprint,
                                 prepared_inputs=prepared_inputs,
                                 scoring='raw values retained; use score_properties with a saved normalizer'))
    print(json.dumps(dict(samples=len(rows), statuses={k:v['status'] for k,v in status.items()}, out=str(out))))
    if args.strict and (failures or any(v['status'] in {'partial', 'unconfigured'} for v in status.values())):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--only', nargs='+')
    p.add_argument('--run-external', action='store_true')
    p.add_argument('--strict', action='store_true')
    collect(p.parse_args())
