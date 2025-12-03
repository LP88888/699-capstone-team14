import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_colwidth", None)

df = pd.read_parquet("data/encoded/combined_raw_datasets_with_cuisine_encoded.parquet")
print(df.head())
print(df.columns)