import json, os, requests, time

HUMANEVAL_PATH = "data/HumanEval.jsonl"
MODEL = "deepseek-r1:latest"  # change if you pulled a different one
OUT_JSONL = "ai_vs_human_pairs.jsonl"

def gen_with_ollama(prompt, model=MODEL, max_tokens=512, temperature=0.2):
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": (
                "You are a Python programming assistant. "
                "Write only valid Python code. Do not include explanations or backticks.\n\n"
                f"{prompt}\n\n# Write the full function implementation below:\n"
            ),
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False
        },
        timeout=300
    )
    resp.raise_for_status()
    return resp.json()["response"]

pairs = []
with open(HUMANEVAL_PATH, "r") as f:
    for line in f:
        task = json.loads(line)
        task_id = task["task_id"]
        prompt = task["prompt"]
        human = task["canonical_solution"]

        # Generate AI code locally
        ai_code = gen_with_ollama(prompt)

        pairs.append({
            "problem_id": task_id,
            "prompt": prompt,
            "human_code": human,
            "ai_code": ai_code
        })
        # small pause is polite for CPU thermals
        time.sleep(0.2)

with open(OUT_JSONL, "w") as wf:
    for row in pairs:
        wf.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved {len(pairs)} pairs to {OUT_JSONL}")
