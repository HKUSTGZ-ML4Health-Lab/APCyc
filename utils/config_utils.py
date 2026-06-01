#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List


def format_args(args: List[str]):
    clean_args = []
    for arg in args:
        if not (arg.startswith('-') or arg.startswith('--')): # value
            clean_args.append(arg)
        else:
            arg = arg.lstrip('-').lstrip('-')
            clean_args.extend(arg.split('='))
    return clean_args


def get_parent_dict(config: dict, key: str):
    key_each_depth = key.split('.')
    for k in key_each_depth[:-1]:
        if k not in config:
            raise KeyError(f'Path key {key} not in the dict')
        config = config[k]
    return config, key_each_depth[-1] # last key

#例子：args = ['training.lr', '0.0001', 'model.name', 'ResNet']
#
def overwrite_values(config:dict, opt_args:List):
    #去掉字符串中的杠杠
    args = format_args(opt_args)
    #奇数是key，偶数是values
    keys, values = args[0::2], args[1::2]
    for key, value in zip(keys, values):
        parent, last_key = get_parent_dict(config, key)
        ori_value = parent[last_key]
        parent[last_key] = type(ori_value)(value)
    return config
'''
例子
config = {
    "training": {
        "lr": 0.001,           # 学习率
        "batch_size": 32,      # 批大小
        "epochs": 100          # 训练轮数
    },
    "model": {
        "name": "UNet",        # 模型名字
        "depth": 4,            # 网络层数
        "features": 64         # 初始通道数
    }
}

'''