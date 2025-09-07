#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a large CSV into fixed-size chunks measured in **data rows** (records),
NOT by text lines. Each part includes the header.

Usage:
  python split_csv_rows.py \
    --in big.csv \
    --out_dir ./parts \
    --rows_per_file 100000 \
    --prefix task_a_train
"""

import csv
import sys
import argparse
import codecs
from pathlib import Path

# ---------- IO helpers ----------

def _safe_open_text(path: Path):
    """Open text file with best-guess encoding (UTF-8/UTF-8-BOM/UTF-16)."""
    with path.open("rb") as rb:
        head = rb.read(4)
    if head.startswith(codecs.BOM_UTF16_LE) or head.startswith(codecs.BOM_UTF16_BE):
        enc = "utf-16"
    elif head.startswith(codecs.BOM_UTF8):
        enc = "utf-8-sig"
    else:
        enc = "utf-8"
    return path.open("r", encoding=enc, newline="")  # newline='' per csv docs

def _strip_nul_iter(f):
    """Yield lines with NULs removed (streaming)."""
    for line in f:
        yield line.replace("\x00", "")

def _sniff_dialect(clean_iter, sample_lines=200):
    sample_chunks = []
    try:
        for _ in range(sample_lines):
            sample_chunks.append(next(clean_iter))
    except StopIteration:
        pass
    sample = "".join(sample_chunks)
    dialect = csv.excel
    try:
        if sample:
            dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        pass
    return dialect

# ---------- Splitter ----------

def split_csv_rows(in_path: Path, out_dir: Path, rows_per_file: int, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    with _safe_open_text(in_path) as fin:
        # Prepare a cleaned iterator for sniffing
        temp_iter = _strip_nul_iter(fin)
        dialect = _sniff_dialect(iter(temp_iter))
        fin.seek(0)

        reader = csv.reader(_strip_nul_iter(fin), dialect=dialect)

        # Read header once
        try:
            header = next(reader)
        except StopIteration:
            print("Input file is empty.", file=sys.stderr)
            return
        if header:
            header[0] = header[0].lstrip("\ufeff")  # strip BOM if present

        part_idx = 1
        data_rows_in_current_part = 0   # counts **data rows** only (excludes header)
        total_data_rows = 0

        def open_part(idx: int):
            part_path = out_dir / f"{prefix}_part{idx:03d}.csv"
            f = part_path.open("w", encoding="utf-8", newline="")
            w = csv.writer(f, dialect=csv.excel)
            w.writerow(header)  # always write header
            return f, w

        fout, writer = open_part(part_idx)

        try:
            for row in reader:
                # When reaching limit, start a new part (write header again)
                if data_rows_in_current_part >= rows_per_file:
                    fout.close()
                    part_idx += 1
                    data_rows_in_current_part = 0
                    fout, writer = open_part(part_idx)

                writer.writerow(row)
                data_rows_in_current_part += 1     # count data row
                total_data_rows += 1

                if total_data_rows % 100_000 == 0:
                    print(f"[progress] {total_data_rows:,} rows -> parts: {part_idx:03d}", file=sys.stderr)
        finally:
            fout.close()

        print(f"[done] total data rows: {total_data_rows:,}", file=sys.stderr)
        print(f"[done] parts written: {part_idx}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Split CSV into chunks by DATA ROWS (records).")
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path")
    ap.add_argument("--out_dir", required=True, help="Directory to write parts")
    ap.add_argument("--rows_per_file", type=int, default=100_000, help="Data rows per part (default: 100000)")
    ap.add_argument("--prefix", default="chunk", help="Output filename prefix (default: 'chunk')")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    split_csv_rows(in_path, out_dir, args.rows_per_file, args.prefix)

if __name__ == "__main__":
    main()
