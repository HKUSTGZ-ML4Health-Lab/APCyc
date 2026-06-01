#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Dict, List, Optional, Tuple

import numpy as np

from evaluation.diversity import diversity


def diversity_consistency(seqs: List[str], ca_structs: List[np.ndarray], seq_th: float, rmsd_th: float) -> Dict[str, Optional[float]]:
    if not seqs:
        return {'div_seq': None, 'div_struct': None, 'co_diversity': None, 'consistency': None}
    if not ca_structs:
        seq_div, _, co_div, cons = diversity(seqs, None)
        return {'div_seq': seq_div, 'div_struct': None, 'co_diversity': co_div, 'consistency': cons}

    lengths = [arr.shape[0] for arr in ca_structs]
    target_len = max(set(lengths), key=lengths.count)
    filtered = [(s, c) for s, c in zip(seqs, ca_structs) if c.shape[0] == target_len]
    if len(filtered) < 2:
        seq_div, _, co_div, cons = diversity(seqs, None)
        return {'div_seq': seq_div, 'div_struct': None, 'co_diversity': co_div, 'consistency': cons}

    seqs_f, cas_f = zip(*filtered)
    cas_arr = np.stack(cas_f, axis=0)
    seq_div, struct_div, co_div, cons = diversity(list(seqs_f), cas_arr)
    return {
        'div_seq': seq_div,
        'div_struct': struct_div,
        'co_diversity': co_div,
        'consistency': cons,
    }
