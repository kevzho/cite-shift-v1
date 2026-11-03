#!/usr/bin/env python3
"""Apply the pre-stated anti-collapse weight-selection rule (JEPA spec change log, 8 Oct 2026).

Choose the SMALLEST weight w for which, at step 300:
  (i)   target_std >= 50% of its step-1 value,
  (ii)  ctx_std    >= 50% of its step-1 value,
  (iii) pred-target cosine at step 300 > cosine at step 50.
Training statistics only; no human label is read. The w = 0 reference is the stopped seed-1 run.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "analysis/results/ml/d2_jepa/vc_tests"


def steps(log):
    out = {}
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith('{"step"'):
            r = json.loads(line)
            out[r["step"]] = r
    return out


def main():
    rows, chosen = [], None
    for w in ("0.001", "0.01", "0.1"):
        log = D / f"jepa_w{w}.log"
        if not log.exists():
            rows.append({"w": w, "status": "not run"})
            continue
        s = steps(log)
        if not all(k in s for k in (1, 50, 300)):
            rows.append({"w": w, "status": f"incomplete (have steps {sorted(s)[:3]}…{sorted(s)[-1:]})"})
            continue
        t_ok = s[300]["target_std"] >= 0.5 * s[1]["target_std"]
        c_ok = s[300]["ctx_std"] >= 0.5 * s[1]["ctx_std"]
        l_ok = s[300]["cos"] > s[50]["cos"]
        ok = t_ok and c_ok and l_ok
        rows.append({"w": w, "target_std_1": s[1]["target_std"], "target_std_300": s[300]["target_std"],
                     "ctx_std_1": s[1]["ctx_std"], "ctx_std_300": s[300]["ctx_std"],
                     "cos_50": s[50]["cos"], "cos_300": s[300]["cos"],
                     "i_target": t_ok, "ii_ctx": c_ok, "iii_learning": l_ok, "qualifies": ok})
        if ok and chosen is None:
            chosen = w
    out = {"rule": __doc__.strip().splitlines()[2:6], "tests": rows, "chosen_w": chosen}
    (D / "selection.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
