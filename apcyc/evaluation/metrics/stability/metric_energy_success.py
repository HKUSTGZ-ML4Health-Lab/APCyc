#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional


def compute_energy_success(rosetta_score_val: Optional[float]) -> bool:
    return rosetta_score_val is not None and rosetta_score_val < 0
