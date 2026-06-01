#!/usr/bin/python
# -*- coding:utf-8 -*-


def attach_failure_counts(target_metrics: dict, failures: dict):
    for key, val in failures.items():
        target_metrics[f'failure.{key}'] = val
