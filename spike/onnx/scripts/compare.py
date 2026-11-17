"""Fidelity report: cosine between Rust/ONNX vectors and the Python worker's vectors.

Usage: <python with numpy> scripts/compare.py <reference.out.jsonl> <candidate.out.jsonl> [more candidates...]
Writes a markdown table to stdout and runs/<candidate>.compare.json next to each candidate.
Retrieval check: for each text request with a "_photo" (media/requests.jsonl), ranks the 16 p*.jpg
photos with both systems and compares top-1 and the top-3 set.
"""
import json
import os
import sys

sys.dont_write_bytecode = True
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(path):
    vectors, errors = {}, {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "vector" in row:
                vectors[row["id"]] = np.asarray(row["vector"], dtype=np.float64)
            else:
                errors[row["id"]] = row.get("error")
    return vectors, errors


with open(os.path.join(ROOT, "media", "requests.jsonl"), encoding="utf-8") as handle:
    requests = [json.loads(line) for line in handle if line.strip()]
kind = {r["id"]: r["kind"] for r in requests}
photos = [r["id"] for r in requests if r["id"].startswith("i_p")]
queries = [r["id"] for r in requests if r["kind"] == "text"]

reference, reference_errors = load(sys.argv[1])
print(f"reference: {os.path.basename(sys.argv[1])} ({len(reference)} vectors, {len(reference_errors)} errors)\n")
print("| candidate | modality | n | min | median | mean | top-1 agree | top-3 set agree |")
print("|--|--|--|--|--|--|--|--|")
for path in sys.argv[2:]:
    candidate, errors = load(path)
    name = os.path.basename(path).replace(".out.jsonl", "")
    report = {"errors": {k: v for k, v in errors.items()}, "cosines": {}, "missing_vs_reference_errors": {}}
    by_kind = {}
    for rid, vector in reference.items():
        if rid in candidate:
            cosine = float(vector @ candidate[rid] / np.linalg.norm(vector) / np.linalg.norm(candidate[rid]))
            report["cosines"][rid] = cosine
            by_kind.setdefault(kind[rid], []).append(cosine)
    for rid in errors:
        if rid in reference_errors:
            report["missing_vs_reference_errors"][rid] = "both failed"
    top1 = top3 = total = 0
    if all(p in candidate and p in reference for p in photos):
        R = np.stack([reference[p] for p in photos])
        C = np.stack([candidate[p] for p in photos])
        for q in queries:
            if q not in candidate or q not in reference:
                continue
            a = np.argsort(-(R @ reference[q]))
            b = np.argsort(-(C @ candidate[q]))
            total += 1
            top1 += int(a[0] == b[0])
            top3 += int(set(a[:3]) == set(b[:3]))
        report["retrieval"] = {"queries": total, "top1_agree": top1, "top3_set_agree": top3}
    for k in ("text", "image", "audio", "video"):
        values = np.array(by_kind.get(k, []))
        if not len(values):
            continue
        retrieval = f"{top1}/{total}" if k == "text" and total else ""
        retrieval3 = f"{top3}/{total}" if k == "text" and total else ""
        print(f"| {name} | {k} | {len(values)} | {values.min():.5f} | {np.median(values):.5f} | {values.mean():.5f} | {retrieval} | {retrieval3} |")
    if errors:
        print(f"| {name} | errors | {len(errors)} | {', '.join(sorted(errors))} | | | | |")
    with open(path.replace(".out.jsonl", ".compare.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
