#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import datetime

#日志等级定义
LEVELS = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR']
LEVELS_MAP = None

#初始化映射关系
def init_map():
    global LEVELS_MAP, LEVELS
    LEVELS_MAP = {}
    for idx, level in enumerate(LEVELS):
        LEVELS_MAP[level] = idx


def get_prio(level):
    global LEVELS_MAP
    if LEVELS_MAP is None:
        init_map()
    return LEVELS_MAP[level.upper()]


def print_log(s, level='INFO', end='\n', no_prefix=False):
    #当前允许输出的最小等级
    pth_prio = get_prio(os.getenv('LOG', 'INFO'))
    #当前日志的等级 
    prio = get_prio(level)

    if prio >= pth_prio:
        if not no_prefix:
            #获取当前时间对象
            now = datetime.datetime.now()
            prefix = now.strftime("%Y-%m-%d %H:%M:%S") + f'::{level.upper()}::'
            print(prefix, end='')
        print(s, end=end)
        #立即刷新缓冲区
        sys.stdout.flush()
