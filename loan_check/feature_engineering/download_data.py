from pathlib import Path
import shutil

import kagglehub



def download_accepted_dataset(dataset):
    raw_path = Path(__file__).resolve().parents[1] / "data" / "raw_data"
    raw_path.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(kagglehub.dataset_download(dataset))

    accepted_file = next(
        (f for f in cache_dir.rglob("*accepted*.csv") if f.is_file()),
        None
    )

    if accepted_file is None:
        raise FileNotFoundError("Accepted file not found")

    destination = raw_path / accepted_file.name
    shutil.copy2(accepted_file, destination)
    return destination


accepted_path = download_accepted_dataset("wordsforthewise/lending-club")
print("Accepted file saved to:", accepted_path)