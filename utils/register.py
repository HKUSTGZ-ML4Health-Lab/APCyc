#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Dict
from copy import deepcopy

_NAMESPACE = {}

#register就是自动给类起名加登记
def register(name):
    def decorator(cls):
        assert name not in _NAMESPACE, f'Class {name} already registered'
        _NAMESPACE[name] = cls
        return cls
    return decorator


def get(name):
    if name not in _NAMESPACE:
        raise ValueError(f'Class {name} not registered')
    return _NAMESPACE[name]

#**kwargs 接收任意数量的关键字参数，以字典形式提供给函数内部
def construct(config: Dict, **kwargs):
    #深拷贝一份配置，避免在函数内修改外部传入的config
    config = deepcopy(config)
    cls_name = config.pop('class')
    cls = get(cls_name)
    config.update(kwargs)
    #返回新的config的类
    return cls(**config)
