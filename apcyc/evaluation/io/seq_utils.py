#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List, Tuple

from evaluation.seq_metric import align_sequences


def seq_identity(seq_a: str, seq_b: str) -> float:
    _, sim = align_sequences(seq_a, seq_b)
    return sim


def pairwise_seq_distance(seqs: List[str]) -> List[List[float]]:
    dists = []
    for i, seq1 in enumerate(seqs):
        row = []
        for j, seq2 in enumerate(seqs):
            _, sim = align_sequences(seq1, seq2)
            row.append(1.0 - sim)
        dists.append(row)
    return dists
