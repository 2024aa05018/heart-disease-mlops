# src/download_data.py
from pathlib import Path
import pandas as pd
import urllib.request
import zipfile
import io

from src.config import RAW_DIR

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

def download_and_extract() -> Path:
    """
    Downloads the UCI Heart Disease ZIP and extracts it into data/raw/.
    Returns the path to the extracted directory.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from: {UCI_ZIP_URL}")
    with urllib.request.urlopen(UCI_ZIP_URL) as resp:
        zip_bytes = resp.read()

    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    z.extractall(RAW_DIR)

    extracted_dir = RAW_DIR / "heart+disease"
    print(f"Extracted to: {extracted_dir}")
    return extracted_dir

def main():
    extracted_dir = download_and_extract()

    # We will use Cleveland data (most commonly used). It may be in different formats.
    # We'll locate a likely file and print directory listing for transparency.
    print("Files extracted:")
    for p in sorted(extracted_dir.glob("*")):
        print(" -", p.name)

if __name__ == "__main__":
    main()
