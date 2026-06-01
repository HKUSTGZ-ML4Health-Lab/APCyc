#!/usr/bin/env python
import os
import re
import sys
from os.path import exists
from typing import List, Tuple

from setuptools import find_namespace_packages, setup


def readme() -> str:
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def get_version() -> str:
    version_file = os.path.join('apcyc', 'version.py')
    namespace = {}
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'), namespace)
    return namespace['__version__']


def parse_requirements(fname: str = 'requirements.txt', with_version: bool = True) -> Tuple[List[str], List[str]]:
    require_fpath = fname

    def parse_line(line):
        if line.startswith('-r '):
            target = line.split(' ')[1]
            absolute_target = os.path.join(os.path.dirname(fname), target)
            for info in parse_require_file(absolute_target):
                yield info
            return

        info = {'line': line}
        if line.startswith('-e '):
            info['package'] = line.split('#egg=')[1]
        else:
            pat = '(' + '|'.join(['>=', '==', '>']) + ')'
            parts = re.split(pat, line, maxsplit=1)
            parts = [p.strip() for p in parts]
            info['package'] = parts[0]
            if len(parts) > 1:
                op, rest = parts[1:]
                if ';' in rest:
                    version, platform_deps = map(str.strip, rest.split(';'))
                    info['platform_deps'] = platform_deps
                else:
                    version = rest
                info['version'] = (op, version)
        yield info

    def parse_require_file(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('http'):
                    continue
                if line and not line.startswith('#') and not line.startswith('--'):
                    for info in parse_line(line):
                        yield info
                elif line and line.startswith('--find-links'):
                    for item in line.split():
                        if 'http' in item:
                            yield dict(dependency_links=item.strip())

    items = []
    deps_link = []
    if exists(require_fpath):
        for info in parse_require_file(require_fpath):
            if 'dependency_links' in info:
                deps_link.append(info['dependency_links'])
                continue
            parts = [info['package']]
            if with_version and 'version' in info:
                parts.extend(info['version'])
            if not sys.version.startswith('3.4') and info.get('platform_deps') is not None:
                parts.append(';' + info['platform_deps'])
            items.append(''.join(parts))
    return items, deps_link


if __name__ == '__main__':
    install_requires, deps_link = parse_requirements('requirements.txt')
    dev_requires, _ = parse_requirements('requirements/dev.txt')

    setup(
        name='apcyc',
        version=get_version(),
        description='Autonomous cyclization and property-aware design for cyclic peptides.',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='Yifan Zhao, Lang Qin, Jintai Chen',
        url='https://github.com/HKUSTGZ-ML4Health-Lab/APCyc',
        keywords=['cyclic peptide', 'protein design', 'diffusion model', 'drug discovery'],
        packages=find_namespace_packages(include=[
            'apcyc*',
            'api*',
            'data*',
            'evaluation*',
            'models*',
            'router*',
            'scripts*',
            'trainer*',
            'utils*',
        ]),
        py_modules=[
            'apcyc_sample',
            'apcyc_train',
            'cal_metrics',
            'data_check',
            'generate',
            'setup_latent_guidance',
            'train',
            'train_router',
        ],
        include_package_data=True,
        package_data={
            'apcyc': ['evaluation/configs/*.yaml'],
            'evaluation': ['dG/*.xml'],
        },
        python_requires='>=3.9',
        license='MIT',
        install_requires=install_requires,
        extras_require={'dev': dev_requires},
        dependency_links=deps_link,
        zip_safe=False)
