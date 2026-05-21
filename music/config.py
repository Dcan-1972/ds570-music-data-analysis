from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

ASSETS_DIR = PROJECT_ROOT / "Assets"
GENRES_DIR = ASSETS_DIR / "Genres"
COUNTRIES_DIR = ASSETS_DIR / "Countries"
FESTIVALS_DIR = ASSETS_DIR / "Festivals"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

GENRE_NAMES = ["Pop", "Rock", "Hip Hop", "Electronic", "Metal"]
