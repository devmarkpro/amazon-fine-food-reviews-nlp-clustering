import os
from typing import TypedDict
import requests
from zipfile import ZipFile
import sys
import logging

logger = logging.getLogger(__name__)


def download_dataset(
    outputdir: str = "./data", show_progress: bool = True
) -> list[str]:

    download_file_name = "archive.zip"
    download_url = (
        "https://www.kaggle.com/api/v1/datasets/download/snap/amazon-fine-food-reviews"
    )
    logger.info(
        f"Downloading dataset from {download_url} to {outputdir}/{download_file_name}"
    )

    response = requests.get(download_url, stream=True)

    os.makedirs(outputdir, exist_ok=True)
    if response.status_code != 200:
        raise requests.exceptions.HTTPError(
            f"Failed to download dataset. Status code: {response.status_code}"
        )

    # Get the total file size for progress tracking
    total_size = int(response.headers.get("content-length", 0))
    downloaded_size = 0

    with open(os.path.join(outputdir, download_file_name), "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded_size += len(chunk)
            if not show_progress:
                continue

            # Display progress
            if total_size > 0:
                progress = (downloaded_size / total_size) * 100
                bar_length = 50
                filled_length = int(bar_length * downloaded_size // total_size)
                bar = "█" * filled_length + "-" * (bar_length - filled_length)
                sys.stdout.write(
                    f"\r[{bar}] {progress:.1f}% ({downloaded_size // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)"
                )
                sys.stdout.flush()
            else:
                # If content-length is not available, show downloaded size only
                sys.stdout.write(f"\rDownloaded: {downloaded_size // 1024 // 1024}MB")
                sys.stdout.flush()

    print()  # New line after progress bar

    logger.info(f"Download completed. Extracting {download_file_name}...")
    with ZipFile(os.path.join(outputdir, download_file_name), "r") as zip_ref:
        zip_ref.extractall(outputdir)

    os.remove(os.path.join(outputdir, download_file_name))
    logger.info("Extraction completed. Dataset is ready for use.")
    # get list of extracted files
    extracted_files = os.listdir(outputdir)
    logger.info("Extracted files:")
    for file in extracted_files:
        logger.info(f"- {file}")

    # return full path of extracted files for further processing if needed
    extracted_files = [os.path.join(outputdir, file) for file in extracted_files]
    return extracted_files


if __name__ == "__main__":
    files = download_dataset()
    for file in files:
        logger.info(f"Downloaded and extracted: {file}")
