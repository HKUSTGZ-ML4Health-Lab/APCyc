#!/usr/bin/python
# -*- coding:utf-8 -*-
from train import parse, main
from utils.logger import print_log
from utils.random_seed import setup_seed, SEED


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(args.seed if hasattr(args, 'seed') else SEED)
    main(args, opt_args)
