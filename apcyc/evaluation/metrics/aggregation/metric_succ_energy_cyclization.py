#!/usr/bin/python
# -*- coding:utf-8 -*-


def compute_succ_energy_cyclization(succ_rows):
    succ_energy_cyclization = any(
        (row.get('cyclization_success') is True) and (row.get('energy_success') is True)
        for row in succ_rows
    )
    return float(succ_energy_cyclization)
