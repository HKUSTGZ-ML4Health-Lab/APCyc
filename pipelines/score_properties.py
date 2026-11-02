"""Property scores with an explicit, reusable normalization population."""
import argparse
import csv
import json
import math
from pathlib import Path
from utils.file_utils import require_distinct_paths

import numpy as np
import pandas as pd

ALIASES = {'rosetta_dG': ['aff_rosetta_dG', 'rosetta_dG'], 'vina_score': ['aff_vina_score', 'vina_score'],
           'logP': ['hydro_logp', 'logP'], 'GRAVY': ['hydro_gravy', 'GRAVY'],
           'TPSA': ['hydro_tpsa', 'TPSA'], 'rTPSA': ['hydro_tpsa_over_heavy_atoms', 'rTPSA', 'TPSA_over_heavy_atoms']}
NORMALIZED = ['rosetta_dG', 'vina_score', 'logP', 'GRAVY', 'TPSA', 'rTPSA']


def validate_rows(df):
    if df.columns.duplicated().any() or 'id' not in df:
        raise ValueError('Property tables require unique column names and an id column.')
    if df.empty or not all(isinstance(sid, str) and sid.strip() for sid in df['id']):
        raise ValueError('Property IDs must be nonempty strings.')
    if df['id'].duplicated().any():
        raise ValueError('Duplicate IDs in raw property table')


def validate_normalizer(normalizer):
    if normalizer.get('format') not in {'apcyc.property_normalizer.v1', 'apcyc.property_normalizer.v2'}:
        raise ValueError('Unsupported property normalizer')
    epsilon = normalizer.get('log_epsilon')
    if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError('Normalizer log_epsilon must be finite and between zero and one.')
    for key in NORMALIZED:
        stats = normalizer.get('statistics', {}).get(key, {})
        count = stats.get('count')
        if type(count) is not int or count < 0:
            raise ValueError(f'{key}: normalizer count must be a nonnegative integer.')
        if count:
            for field in ['mean', 'std']:
                value = stats.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'{key}: normalizer {field} must be finite.')
            if stats['std'] < 0:
                raise ValueError(f'{key}: normalizer std must be nonnegative.')


def numeric(df, key):
    for name in ALIASES.get(key, [key]):
        if name in df:
            return pd.to_numeric(df[name], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return pd.Series(np.nan, index=df.index, dtype=float)


def valid_affinity(df, key, legacy_masks=False):
    x = numeric(df, key)
    if legacy_masks and key == 'rosetta_dG':
        x = x.where(x <= 10)
    elif legacy_masks and key == 'vina_score':
        x = x.where(x <= 0)
    return x


def fit_normalizer(df, population):
    validate_rows(df)
    selected = df[df['id'].isin(population)]
    if selected.empty:
        raise ValueError('No property rows match the requested normalization population')
    stats = {}
    for key in NORMALIZED:
        x = valid_affinity(selected, key).dropna()
        stats[key] = dict(count=len(x), mean=float(x.mean()) if len(x) else None,
                          std=float(x.std(ddof=0)) if len(x) else None)
    return dict(format='apcyc.property_normalizer.v2', fit_ids=sorted(selected['id'].tolist()),
                missing_population_ids=sorted(set(population) - set(selected['id'])), statistics=stats,
                affinity_masks={}, log_epsilon=1e-12)


def score(df, normalizer):
    validate_rows(df)
    validate_normalizer(normalizer)
    def z(key):
        x = valid_affinity(df, key, legacy_masks=normalizer['format'] == 'apcyc.property_normalizer.v1')
        stats = normalizer['statistics'][key]
        if not stats['count']:
            return x * np.nan
        return x * 0 if stats['std'] == 0 else (x - stats['mean']) / stats['std']
    out = df[[c for c in ['id', 'sequence'] if c in df]].copy()
    out['solubility_camsol_score'] = numeric(df, 'camsol_score')
    out['rosetta_dG'], out['vina_score'] = numeric(df, 'rosetta_dG'), numeric(df, 'vina_score')
    out['rosetta_total_score'] = numeric(df, 'rosetta_total_score')
    out['affinity'] = -(z('rosetta_dG') + z('vina_score')) / 2
    out['cycpep_proteolysis_sumscore'] = -numeric(df, 'cycpep_proteolysis_sumscore')
    out['Pproxy'] = z('logP') + z('GRAVY') - z('TPSA') - z('rTPSA')
    eps = normalizer['log_epsilon']
    el, im = numeric(df, 'immune_el'), numeric(df, 'immune_im')
    # Preserve missing values. Negative probabilities are invalid, not epsilon-clipped successes.
    out['Sfinal'] = -(np.log(el.where(el >= 0).clip(lower=eps)) + np.log(im.where(im >= 0).clip(lower=eps)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--normalizer', type=Path)
    mode.add_argument('--fit-ids', type=Path, help='Training IDs only; writes --save-normalizer.')
    mode.add_argument('--legacy-cohort-normalization', action='store_true', help='Explicitly reproduce historical per-table z-scoring.')
    p.add_argument('--save-normalizer', type=Path)
    args = p.parse_args()
    if args.normalizer and args.save_normalizer:
        p.error('--save-normalizer is only used when fitting statistics.')
    try:
        require_distinct_paths([args.output, args.save_normalizer],
                               [args.input, args.fit_ids, args.normalizer])
    except ValueError as error:
        p.error(str(error))
    with args.input.open(newline='') as handle:
        header = next(csv.reader(handle), [])
    if not header or len(header) != len(set(header)):
        p.error('Input CSV requires unique column names.')
    df = pd.read_csv(args.input, dtype={'id': str}, keep_default_na=False)
    if args.normalizer:
        norm = json.loads(args.normalizer.read_text())
    else:
        ids = set(args.fit_ids.read_text().splitlines()) if args.fit_ids else set(df['id'])
        norm = fit_normalizer(df, ids)
        if not args.save_normalizer:
            p.error('Fitting requires --save-normalizer so scores can be reproduced.')
        args.save_normalizer.parent.mkdir(parents=True, exist_ok=True)
        args.save_normalizer.write_text(json.dumps(norm, indent=2) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    score(df, norm).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
