#!/usr/bin/env python3
"""Audit or reproducibly reconstruct a split from an explicit FoldSeek cluster map."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random


def read_clusters(path):
    groups, owner = {}, {}
    with path.open() as stream:
        rows = csv.DictReader(stream, delimiter='\t')
        if not {'cluster_center', 'member'} <= set(rows.fieldnames or []):
            raise ValueError('Cluster TSV needs cluster_center and member columns')
        for row in rows:
            cluster, member = row['cluster_center'].strip(), row['member'].strip()
            if not cluster or not member or member in owner:
                raise ValueError('Empty or duplicate cluster membership')
            owner[member] = cluster
            groups.setdefault(cluster, []).append(member)
    if not owner:
        raise ValueError('Empty cluster map')
    return groups, owner


def audit(owner, train, valid):
    missing = sorted((train | valid) - owner.keys())
    if missing:
        raise ValueError(f'{len(missing)} split IDs lack cluster membership')
    overlap = sorted({owner[x] for x in train} & {owner[x] for x in valid})
    return dict(train_samples=len(train), valid_samples=len(valid),
                shared_sample_ids=sorted(train & valid), shared_clusters=overlap,
                unused_cluster_members=len(owner.keys() - (train | valid)),
                cluster_independent=not overlap and not (train & valid))


def read_ids(path):
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate split IDs')
    return set(ids)


def run(args):
    groups, owner = read_clusters(args.cluster_map)
    if args.train_ids or args.valid_ids:
        if not args.train_ids or not args.valid_ids:
            raise ValueError('Audit mode requires both --train-ids and --valid-ids')
        train, valid = read_ids(args.train_ids), read_ids(args.valid_ids)
        mode = 'audit_existing'
    else:
        if args.seed is None:
            raise ValueError('Reconstruction requires an explicit --seed; the paper seed is unknown')
        if not 0 < args.valid_clusters < len(groups):
            raise ValueError('Validation cluster count must leave both splits nonempty')
        selected = set(random.Random(args.seed).sample(sorted(groups), args.valid_clusters))
        valid = {x for c in selected for x in groups[c]}
        train = set(owner) - valid
        mode = 'new_seeded_split_not_historical_recovery'
    index_rows = None
    if args.mmap_index:
        index_rows = {}
        for line in args.mmap_index.read_text().splitlines():
            key = line.split('\t')[0]
            if key in index_rows:
                raise ValueError('Duplicate mmap index ID')
            index_rows[key] = line
        if set(index_rows) - (train | valid):
            raise ValueError('Mmap index contains IDs outside the selected split')
        excluded = sorted((train | valid) - index_rows.keys())
        train &= index_rows.keys(); valid &= index_rows.keys()
    else:
        excluded = []
    report = audit(owner, train, valid)
    report.update(mode=mode, seed=args.seed if mode.startswith('new') else None,
                  total_clusters=len(groups), requested_valid_clusters=args.valid_clusters,
                  excluded_without_mmap_record=excluded,
                  cluster_map_sha256=hashlib.sha256(args.cluster_map.read_bytes()).hexdigest())
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, ids in [('train', train), ('valid', valid)]:
        (args.out_dir/f'{name}_ids.txt').write_text(''.join(x+'\n' for x in sorted(ids)))
        if index_rows is not None:
            (args.out_dir/f'{name}_index.txt').write_text(''.join(index_rows[x]+'\n' for x in sorted(ids)))
    (args.out_dir/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    if not report['cluster_independent']:
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cluster-map', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--train-ids', type=Path)
    p.add_argument('--valid-ids', type=Path)
    p.add_argument('--mmap-index', type=Path)
    p.add_argument('--seed', type=int)
    p.add_argument('--valid-clusters', type=int, default=616)
    run(p.parse_args())
