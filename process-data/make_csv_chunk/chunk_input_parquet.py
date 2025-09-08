#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a Parquet file into multiple CSV files by DATA ROW COUNTS.

Default:
  --sizes 100000,100000,100000,100000,100000,100000,40000
which yields 7 CSVs totaling 640k rows.

If you prefer equal chunks:
  --rows_per_file 100000
and it will make N full 100k CSVs plus a final remainder CSV automatically.

Requires: pyarrow>=8

Usage examples:
  python parquet_to_csv_chunks.py \
    --in task_a_test_set_sample.parquet \
    --out_dir ./csv_parts \
    --prefix task_a_test \
    --sizes 100000,100000,100000,100000,100000,100000,40000

  # or equal-sized chunks + remainder:
  python parquet_to_csv_chunks.py \
    --in task_a_test_set_sample.parquet \
    --out_dir ./csv_parts \
    --prefix task_a_test \
    --rows_per_file 100000
"""

import os
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pacsv

def _write_csv(table: pa.Table, path: Path):
    # Ensure parent exists
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write CSV with header (default)
    pacsv.write_csv(table, path)

def _next_part_path(out_dir: Path, prefix: str, idx: int) -> Path:
    return out_dir / f"{prefix}_part{idx:03d}.csv"

def split_parquet_by_sizes(in_path: Path, out_dir: Path, prefix: str, sizes: list[int], batch_size: int = 100_000):
    pf = pq.ParquetFile(in_path)
    schema = pf.schema_arrow
    total_rows = pf.metadata.num_rows

    # Sanity: sizes should not exceed total_rows
    if sum(sizes) > total_rows:
        raise ValueError(f"Sum of sizes ({sum(sizes):,}) exceeds parquet rows ({total_rows:,}).")

    part_idx = 1
    need = sizes[0] if sizes else 0
    acc_tables = []          # list of pa.Table chunks for current part
    acc_rows = 0             # rows accumulated for current part
    sizes_pos = 0
    written_rows = 0

    # Stream through parquet in batches
    for batch in pf.iter_batches(batch_size=batch_size):
        tbl = pa.Table.from_batches([batch])  # table view of this batch
        start = 0
        while start < tbl.num_rows:
            if sizes_pos >= len(sizes):
                # We've written all requested parts; ignore any remaining rows
                start = tbl.num_rows
                break

            if need <= 0:
                # Current part complete -> write it
                out_path = _next_part_path(out_dir, prefix, part_idx)
                _write_csv(pa.concat_tables(acc_tables) if acc_tables else pa.Table.from_arrays(
                    [pa.array([], type=col.type) for col in schema], names=[f.name for f in schema]
                ), out_path)
                part_idx += 1
                acc_tables = []
                acc_rows = 0
                sizes_pos += 1
                if sizes_pos >= len(sizes):
                    break
                need = sizes[sizes_pos]

            # Fill current part from the remaining rows in this batch
            remaining_in_batch = tbl.num_rows - start
            take = min(need, remaining_in_batch)
            if take > 0:
                acc_tables.append(tbl.slice(start, take))
                start += take
                acc_rows += take
                written_rows += take
                need -= take

    # Flush the final part if there’s anything pending
    if acc_tables and sizes_pos < len(sizes):
        out_path = _next_part_path(out_dir, prefix, part_idx)
        _write_csv(pa.concat_tables(acc_tables), out_path)
        part_idx += 1
        sizes_pos += 1

    # If sizes had more parts than rows available, write empty CSVs with only headers
    while sizes_pos < len(sizes):
        out_path = _next_part_path(out_dir, prefix, part_idx)
        empty = pa.Table.from_arrays(
            [pa.array([], type=col.type) for col in schema], names=[f.name for f in schema]
        )
        _write_csv(empty, out_path)
        part_idx += 1
        sizes_pos += 1

    print(f"[done] parquet rows: {total_rows:,} | requested: {sum(sizes):,} | written: {written_rows:,}")
    print(f"[done] parts written: {part_idx-1}")

def split_parquet_equal_chunks(in_path: Path, out_dir: Path, prefix: str, rows_per_file: int, batch_size: int = 100_000):
    pf = pq.ParquetFile(in_path)
    total_rows = pf.metadata.num_rows
    full = total_rows // rows_per_file
    rem = total_rows % rows_per_file
    sizes = [rows_per_file] * full + ([rem] if rem else [])
    if not sizes:
        sizes = [0]
    split_parquet_by_sizes(in_path, out_dir, prefix, sizes, batch_size=batch_size)

def main():
    ap = argparse.ArgumentParser(description="Split Parquet -> multiple CSVs by record count.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input .parquet path")
    ap.add_argument("--out_dir", required=True, help="Output directory for CSV parts")
    ap.add_argument("--prefix", default="chunk", help="Output filename prefix (default: 'chunk')")
    ap.add_argument("--sizes", default="", help="Comma-separated list of part sizes, e.g. 100000,100000,40000")
    ap.add_argument("--rows_per_file", type=int, default=0, help="Equal rows per file (ignored if --sizes provided)")
    ap.add_argument("--batch_size", type=int, default=100_000, help="Parquet scan batch size (default: 100k)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)

    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    if args.sizes.strip():
        sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    else:
        # Default to 6×100k + 1×40k to cover 640k rows
        if args.rows_per_file > 0:
            return split_parquet_equal_chunks(in_path, out_dir, args.prefix, args.rows_per_file, batch_size=args.batch_size)
        sizes = [100_000]*6 + [40_000]

    split_parquet_by_sizes(in_path, out_dir, args.prefix, sizes, batch_size=args.batch_size)

if __name__ == "__main__":
    main()

# python chunk_input_parquet.py \
#   --in ../../data/SemEval-2026-Task13/task_a/task_a_training_set_1.parquet \
#   --out_dir ../../data/SemEval-2026-Task13/task_a/chunks \
#   --sizes 10000,10000,10000,10000,10000,50000,50000,50000,50000,50000,50000,50000,50000,50000