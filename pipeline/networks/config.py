from pathlib import Path as PATH

CLEANED_DATA_PATH = PATH("data/encoded/combined_raw_datasets_with_cuisine_clean_encoded.parquet")
DATA_PATH = PATH("data/encoded/combined_raw_datasets_with_cuisine_encoded.parquet")
PHASE_1_REPORTS_PATH = PATH("reports/phase1")
PHASE_1_VIZ_PATH = PHASE_1_REPORTS_PATH / "viz"
PHASE_2_REPORTS_PATH = PATH("reports/phase2")
PHASE_2_VIZ_PATH = PHASE_2_REPORTS_PATH / "viz"

def setup_paths():
    PHASE_1_REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    PHASE_1_VIZ_PATH.mkdir(parents=True, exist_ok=True)
    PHASE_2_REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    PHASE_2_VIZ_PATH.mkdir(parents=True, exist_ok=True)