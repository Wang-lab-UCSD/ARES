"""5x MiniMax-M3 Tier-2 mechanism vote on the new-prompt consensus.
Each resolved pair is classified 5 independent times; majority leaf wins.
Abstain-aware: NEED_HUMAN / ERR / NO_REPORT don't count as votes; assign a leaf only
if a UNIQUE non-abstain leaf wins >=2 votes, else NEED_HUMAN_LABEL_<dichotomy>.
"""
from __future__ import annotations
import csv, sys, argparse, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from common import resolve_dir
from deterministic import extract_deterministic, load_report
from llm_mechanism import mechanism_classify, GROUPS

SRC = "analysis/dichotomy_newprompt_consensus.csv"
OUT = "analysis/dichotomy_newprompt_mechanism_vote.csv"

def classify_once(row):
    cell = row["cell"]; t = str(row["tf_a"]).upper(); p = str(row["tf_b"]).upper()
    dic = row["consensus"]; sub = row.get("protein_subtype", "") or ""
    if dic not in GROUPS:
        return dict(mechanism="NA_UNRESOLVED", key_factor="", evidence="", mechanism_freeform="")
    d = resolve_dir(cell, t, p)
    if not d:
        return dict(mechanism="NO_REPORT", key_factor="", evidence="", mechanism_freeform="")
    try:
        det = extract_deterministic(f"{cell}_{t}_{p}", d)
        o = mechanism_classify(load_report(d), det, dic, sub, provider="minimax")
        m = o["mechanism"]
        o["mechanism"] = f"NEED_HUMAN_LABEL_{dic}" if m.startswith("INVALID:") else m
        return o
    except Exception as e:
        return dict(mechanism=f"ERR:{str(e)[:20]}", key_factor="", evidence="", mechanism_freeform="")

def vote(leaves, dic):
    bad = lambda v: (not v) or v.startswith(("NEED_HUMAN", "ERR", "NO_REPORT", "NA_"))
    na = [v for v in leaves if not bad(v)]
    if not na:
        return f"NEED_HUMAN_LABEL_{dic}"
    cnt = collections.Counter(na); top, n = cnt.most_common(1)[0]
    ties = [k for k, c in cnt.items() if c == n]
    return top if (len(ties) == 1 and n >= 2) else f"NEED_HUMAN_LABEL_{dic}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvote", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--group", default="", help="restrict to one dichotomy group, e.g. PROTEIN")
    ap.add_argument("--restrict-pairs", default="", help="csv (cell,tf_a,tf_b) to restrict the re-vote to")
    ap.add_argument("--out", default=OUT, help="output csv path")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(SRC)))
    resolved = [r for r in rows if r["consensus"] in GROUPS]
    if a.group:
        resolved = [r for r in resolved if r["consensus"] == a.group]
    if a.restrict_pairs:
        keep = {(r["cell"], str(r["tf_a"]).upper(), str(r["tf_b"]).upper())
                for r in csv.DictReader(open(a.restrict_pairs))}
        resolved = [r for r in resolved
                    if (r["cell"], str(r["tf_a"]).upper(), str(r["tf_b"]).upper()) in keep]
    print(f"{len(rows)} pairs; {len(resolved)} resolved -> {a.nvote}x MiniMax-M3 vote ({a.jobs} parallel)", flush=True)
    tasks = [(i, r) for i, r in enumerate(resolved) for _ in range(a.nvote)]
    results = collections.defaultdict(list)
    done = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for i, o in ex.map(lambda t: (t[0], classify_once(t[1])), tasks):
            results[i].append(o); done += 1
            if done % 500 == 0:
                print(f"  ...{done}/{len(tasks)} calls", flush=True)
    out = []
    for i, r in enumerate(resolved):
        outs = results[i]; leaves = [o["mechanism"] for o in outs]
        win = vote(leaves, r["consensus"])
        rep = next((o for o in outs if o["mechanism"] == win), outs[0])
        out.append(dict(cell=r["cell"], tf_a=r["tf_a"], tf_b=r["tf_b"], dichotomy=r["consensus"],
                        reader=r.get("reader", ""), actor=r.get("actor", ""), mechanism=win,
                        votes=";".join(leaves), key_factor=rep.get("key_factor", ""),
                        evidence=rep.get("evidence", "")))
    print("mechanism vote results:", dict(collections.Counter(o["mechanism"] for o in out).most_common(25)))
    nhl = sum(o["mechanism"].startswith("NEED_HUMAN") for o in out)
    print(f"NEED_HUMAN (no majority): {nhl}/{len(out)}")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell","tf_a","tf_b","dichotomy","reader","actor",
                                          "mechanism","votes","key_factor","evidence"])
        w.writeheader(); w.writerows(out)
    print("saved ->", a.out)

if __name__ == "__main__":
    main()
