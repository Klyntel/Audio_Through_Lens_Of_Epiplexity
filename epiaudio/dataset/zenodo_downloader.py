import requests
import zipfile
from tqdm import tqdm
from pathlib import Path

def download_zenodo(record_id: str, output_dir: str="", do_not_download: list[str] | None=None) -> None:
    """Downloads a (public) record form Zenodo. You can specify that certain files should not be
    downloaded with the do_not_download argument."""
    if do_not_download is None:
        do_not_download = []

    url = f"https://zenodo.org/api/records/{record_id}"
    record = requests.get(url).json()

    output_path = Path(output_dir) if output_dir else Path(output_dir) / "data" / f"zenodo_{record_id}"
    output_path.mkdir(parents=True, exist_ok=True)

    for file_info in record["files"]:
        filename = file_info["key"]
        dest = output_path / filename

        if Path(filename).stem in do_not_download:
            continue

        # A zip is considered done once its extraction dir exists (the zip
        # itself is deleted after extraction); other files once they exist.
        extract_dir = output_path / Path(filename).stem
        if filename.endswith(".zip") and extract_dir.exists():
            print(f"Skipping {filename} (already extracted)...")
            continue
        if dest.exists():
            print(f"Skipping {filename} (already downloaded)...")
            continue

        print(f"Downloading {filename}...")
        with requests.get(file_info["links"]["self"], stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=filename) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

        if filename.endswith(".zip"):
            extract_dir = output_path / Path(filename).stem
            if not extract_dir.exists():
                print(f"Extracting {filename} -> {extract_dir}/")
                with zipfile.ZipFile(dest, "r") as zf:
                    zf.extractall(extract_dir)
            dest.unlink()
