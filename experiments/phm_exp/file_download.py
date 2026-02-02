"""
Python script to test the download of a file
"""

import logging
import gdown
from pathlib import Path

URL = "https://drive.google.com/uc?id=14MTmAExKfne5il1MJAPkOo04yv5UMtH4"

logger = logging.getLogger(__name__)

def download(url: str, path: Path):
    logger.info("Downloading dataset...")
    gdown.download(url, str(path / "downloaded_txt.txt"), quiet=False)

experiment_path = Path.cwd()

print("-"*50)
print("Downloading file")
print("-"*50)

download(url=URL,path=experiment_path)
