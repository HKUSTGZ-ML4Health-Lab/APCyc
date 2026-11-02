#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List
import yaml


def format_args(args: List[str]):
    pairs = []
    index = 0
    while index < len(args):
        key = args[index].lstrip('-')
        if '=' in key:
            key, value = key.split('=', 1)
            index += 1
        else:
            if index + 1 >= len(args):
                raise ValueError(f'Missing override value for {key}')
            value = args[index + 1]
            index += 2
        if not key:
            raise ValueError('Empty configuration override key')
        pairs.extend([key, value])
    return pairs


def get_parent_dict(config: dict, key: str):
    key_each_depth = key.split('.')
    for k in key_each_depth[:-1]:
        if not isinstance(config, dict) or k not in config:
            raise KeyError(f'Path key {key} not in the dict')
        config = config[k]
    if not isinstance(config, dict) or key_each_depth[-1] not in config:
        raise KeyError(f'Unknown configuration path: {key}')
    return config, key_each_depth[-1] # last key

#
def overwrite_values(config:dict, opt_args:List):
    #去掉字符串中的杠杠
    args = format_args(opt_args)
    #奇数是key，偶数是values
    keys, values = args[0::2], args[1::2]
    for key, value in zip(keys, values):
        parent, last_key = get_parent_dict(config, key)
        ori_value = parent[last_key]
        parsed = value if isinstance(ori_value, str) else yaml.safe_load(value)
        if ori_value is not None and type(parsed) is not type(ori_value):
            if isinstance(ori_value, float) and type(parsed) is int:
                parsed = float(parsed)
            else:
                raise ValueError(f'Override {key} must be {type(ori_value).__name__}, got {parsed!r}')
        parent[last_key] = parsed
    return config
