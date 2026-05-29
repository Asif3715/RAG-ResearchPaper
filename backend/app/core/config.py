from pathlib import Path


def resolve_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


DATA_DIR = resolve_data_dir()
UPLOADS_DIR = DATA_DIR / "uploads"
EXTRACTED_DIR = DATA_DIR / "extracted"
