#!/usr/bin/python
# -*- coding:utf-8 -*-


def compute_succ_cyclization(succ_rows):
    succ_cyclization = any(row.get('cyclization_success') is True for row in succ_rows)
    return float(succ_cyclization)
