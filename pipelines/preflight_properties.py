"""Check property-stage configuration and dependencies without submitting any sequences."""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path
import yaml

from pipelines.collect_properties import expand, sha256


def inspect(config_path, selected=None):
    config = yaml.safe_load(config_path.read_text())
    root = Path(__file__).resolve().parents[1]
    context = dict(repo=str(root), output='/preflight', inputs='/preflight/inputs',
                   fasta='/preflight/sequences.fasta', structures='/preflight/structure_zips',
                   pdb_list='/preflight/pdb_list.txt', stage_dir='/preflight/stage',
                   stage_output='/preflight/stage/results.csv', chain=config.get('chain', 'L'))
    if selected and set(selected) - (set(config['stages']) | {'basic'}):
        raise ValueError('Unknown stage in --only')
    reports = {'basic': dict(status='ready', missing_environment_names=[], errors=[], dependency_sha256={})}
    for name, spec in config['stages'].items():
        if selected and name not in selected:
            continue
        if spec.get('import'):
            errors, dependencies = [], {}
            try:
                path = Path(expand(spec['import'], context)).expanduser()
                path = path if path.is_absolute() else config_path.resolve().parent / path
                dependencies[str(path)] = sha256(path)
                with path.open(newline='') as handle:
                    header = csv.DictReader(handle, delimiter=spec.get('delimiter', ',')).fieldnames
                required = {spec.get('id_column', 'id')} | set(spec['columns'].values())
                if not header or len(header) != len(set(header)) or not required <= set(header):
                    raise ValueError('Imported CSV requires unique headers and all configured columns.')
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
            reports[name] = dict(status='blocked' if errors else 'ready', mode='import',
                                 missing_environment_names=[], errors=errors, dependency_sha256=dependencies)
            continue
        if not spec.get('enabled', False):
            reports[name] = dict(status='disabled', missing_environment_names=[], errors=[], dependency_sha256={})
            continue
        missing = [k for k in spec.get('required_env', []) if not os.environ.get(k)]
        dependencies, errors = {}, []
        if not missing:
            for value in spec.get('dependencies', []):
                try:
                    path = Path(expand(value, context))
                    dependencies[path.name] = sha256(path)
                except Exception as exc:
                    errors.append(f'{type(exc).__name__}: {exc}')
            try:
                executable = expand(spec['argv'][0], context)
                if shutil.which(executable) is None:
                    errors.append('Configured stage executable is missing or not executable')
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
        reports[name] = dict(status='ready' if not missing and not errors else 'blocked',
                             missing_environment_names=missing, errors=errors,
                             dependency_sha256=dependencies)
    return dict(stages=reports, ready=all(r['status'] in {'ready', 'disabled'} for r in reports.values()),
                configuration_sha256=sha256(config_path),
                scope='file/executable checks only; availability, authentication and model inference require a run')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--only', nargs='+')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    result = inspect(a.config, a.only)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['ready'] else 1)
