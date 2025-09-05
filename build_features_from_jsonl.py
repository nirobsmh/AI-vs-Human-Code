# build_features_from_jsonl.py
import argparse, json, pandas as pd
from features_lib import extract_features_for_row

def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "human_code" in row:
                rows.append({"problem_id": row["problem_id"], "prompt": row["prompt"], "code": row["human_code"], "label": "human"})
            if "ai_code" in row:
                rows.append({"problem_id": row["problem_id"], "prompt": row["prompt"], "code": row["ai_code"], "label": "ai"})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ai_vs_human_code.jsonl")
    ap.add_argument("--out_csv", default="features.csv")
    ap.add_argument("--out_parquet", default="features.parquet")
    ap.add_argument("--keep_text", action="store_true")
    args = ap.parse_args()

    df = load_jsonl(args.input)
    feat_rows = []
    for _, r in df.iterrows():
        feats = extract_features_for_row(r["code"] or "")
        feats["problem_id"] = r["problem_id"]
        feats["label"] = r["label"]
        if args.keep_text:
            feats["prompt"] = r["prompt"]
            feats["code"] = r["code"]
        feat_rows.append(feats)

    out = pd.DataFrame(feat_rows)
    out.to_csv(args.out_csv, index=False)
    out.to_parquet(args.out_parquet, index=False)
    print(f"✅ Saved {len(out)} rows -> {args.out_csv}, {args.out_parquet}")

if __name__ == "__main__":
    main()
