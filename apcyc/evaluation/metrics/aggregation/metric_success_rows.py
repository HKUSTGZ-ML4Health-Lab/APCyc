#!/usr/bin/python
# -*- coding:utf-8 -*-


def select_success_rows(metrics_per_sample, samples, k_limit):
    if k_limit is None or k_limit <= 0:
        return metrics_per_sample
    if all('sample_idx' in s for s in samples):
        sorted_samples = sorted(samples, key=lambda s: int(s.get('sample_idx', 0)))
    else:
        sorted_samples = samples
    succ_candidates = set()
    for s in sorted_samples[: int(k_limit)]:
        succ_candidates.add((s.get('sample_id'), s.get('sample_idx')))
    return [
        row for row in metrics_per_sample
        if (row.get('sample_id'), row.get('sample_idx')) in succ_candidates
    ]
