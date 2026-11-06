"""Compare two identical-config runs episode by episode (all fields except wall-clock seconds)."""
import sys, pandas as pd
a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
cols = [c for c in a.columns if c != "seconds"]
assert len(a) == len(b)
diff = (a[cols] != b[cols]).any(axis=1)
print(f"episodes={len(a)} identical_rows={int((~diff).sum())} differing_rows={int(diff.sum())}")
print("success identical:", bool((a.success == b.success).all()), "| max_reward identical:", bool((a.max_reward == b.max_reward).all()),
      "| mean_abs_d2 max abs diff:", float((a.mean_abs_d2 - b.mean_abs_d2).abs().max()))
print(a[["episode", "success", "max_reward", "steps", "mean_abs_d2"]].to_string(index=False))
if diff.any(): print(pd.concat([a[diff][cols], b[diff][cols]]).to_string())
