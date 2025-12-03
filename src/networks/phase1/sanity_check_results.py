from pathlib import Path
import json
import pandas as pd

OUT = Path("reports/modeling")

# 1) High-level summary
with open(OUT / "summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)

print("Summary:")
for k, v in summary.items():
    print(f"{k}: {v}")

# 2) Peek at classification report
rep = pd.read_csv(OUT / "classification_report_tfidf_only.csv")
print(rep.head())