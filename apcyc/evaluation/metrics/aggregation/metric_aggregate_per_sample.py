#!/usr/bin/python
# -*- coding:utf-8 -*-
from apcyc.evaluation.io import aggregate


def aggregate_per_sample_metrics(
    metrics_per_sample,
    aggregation_modes,
    validity_filter,
    metric_direction,
):
    per_sample_keys = [
        'ramachandran_allowed',
        'ramachandran_favored',
        'energy_success',
        'bond_length_validity',
        'rosetta_interface_dg',
        'rosetta_total_score',
        'vina_score',
    ]
    aggregated = {}
    for mode in aggregation_modes:
        for key in per_sample_keys:
            values = []
            for row in metrics_per_sample:
                if validity_filter and key not in ('cyclization_success',) and not row.get('valid'):
                    continue
                if key != 'cyclization_success' and not row.get('energy_success'):
                    continue
                val = row.get(key)
                if isinstance(val, bool):
                    val = float(val)
                values.append(val)
            if mode == 'best_of_k':
                agg_val = aggregate.best_of_k(values, metric_direction(key))
            else:
                agg_val = aggregate.mean_over_valid(values)
            aggregated[f'{mode}.{key}'] = agg_val
    return aggregated
