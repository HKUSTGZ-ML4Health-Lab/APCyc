#!/usr/bin/env python3
"""Export a self-contained frozen scorer from an AE, reconstructed context and heads."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--autoencoder', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--head', action='append', required=True, metavar='TARGET=PATH')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    heads = {}
    for item in args.head:
        key, separator, path = item.partition('=')
        if not separator or not key or not path or key in heads:
            parser.error('--head requires distinct TARGET=PATH entries')
        heads[key] = Path(path)
    from apcyc.property_scorer import export_scorer
    result = export_scorer(args.autoencoder, args.context, heads, args.out)
    print(f"Exported {len(result['heads'])} frozen heads: {args.out}")


if __name__ == '__main__':
    main()
