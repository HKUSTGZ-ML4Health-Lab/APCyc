from pathlib import Path


def main():
    for path in sorted(Path('.').rglob('*.py')):
        if any(part in {'.git', '__pycache__', 'build', 'dist'} for part in path.parts):
            continue
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    print('syntax ok')


if __name__ == '__main__':
    main()
