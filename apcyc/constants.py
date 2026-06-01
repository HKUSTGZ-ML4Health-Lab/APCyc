#!/usr/bin/python
# -*- coding:utf-8 -*-

CYCLE_TYPES = ['HT', 'SS', 'ISO']
TYPE2IDX = {name: i for i, name in enumerate(CYCLE_TYPES)}
IDX2TYPE = {i: name for name, i in TYPE2IDX.items()}
