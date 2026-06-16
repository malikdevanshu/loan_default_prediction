import shutil

from pathlib import Path

import kagglehub

from config.config import PROJECT_ROOT


def download_accepted_dataset(dataset: str) -> Path:
    raw_path = PROJECT_ROOT / "data" / "raw_data"
    raw_path.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(kagglehub.dataset_download(dataset))

    accepted_file = next(
        (f for f in cache_dir.rglob("*accepted*.csv") if f.is_file()), None
    )

    if accepted_file is None:
        raise FileNotFoundError("Accepted file not found")

    destination = raw_path / accepted_file.name
    shutil.copy2(accepted_file, destination)
    return destination


if __name__ == "__main__":
    accepted_path = download_accepted_dataset("wordsforthewise/lending-club")
    print("Accepted file saved to:", accepted_path)
