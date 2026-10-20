#!/usr/bin/env python3
"""Check relative inline Markdown links/images and heading anchors in public docs.

Offline only; does not check external URLs or implement a full Markdown parser.
"""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def without_fences(text):
    return re.sub(r'^```[^\n]*\n.*?^```\s*$', '', text, flags=re.M | re.S)


def headings(path):
    counts = {}
    anchors = set()
    for title in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', without_fences(path.read_text()), re.M):
        slug = re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-')
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        anchors.add(f'{slug}-{count}' if count else slug)
    return anchors


def main():
    paths = [ROOT / 'README.md', ROOT / 'experiments/canonical70k/README.md']
    paths.extend(sorted((ROOT / 'docs').rglob('*.md')))
    failures = []
    checked = 0
    for path in paths:
        text = without_fences(path.read_text())
        for href in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)', text):
            parts = urlsplit(href)
            if parts.scheme or parts.netloc:
                continue
            checked += 1
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if not target.exists():
                failures.append(f'{path.relative_to(ROOT)}: missing {href}')
            elif parts.fragment and target.suffix == '.md':
                if unquote(parts.fragment) not in headings(target):
                    failures.append(f'{path.relative_to(ROOT)}: missing anchor {href}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'PASS: {checked} local links/images/anchors in {len(paths)} Markdown files')


if __name__ == '__main__':
    main()
