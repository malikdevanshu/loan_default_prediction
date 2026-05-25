import kagglehub 
from pathlib import Path
import shutil

def download_dataset_from_kagglehub(dataset):
    raw_path = Path(__file__).resolve().parents[1] / "data" / "raw_data"
    raw_path.mkdir(parents= True, exist_ok= True)

    cache_dir = Path(kagglehub.dataset_download(dataset))

    for item in cache_dir.iterdir():
        target = raw_path / item.name

        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    return raw_path

raw_path = download_dataset_from_kagglehub("wordsforthewise/lending-club")
print("Files are in:", raw_path)            