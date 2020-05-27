import os
import urllib.request
import zipfile


BASE_DIR = os.environ.get("BASE_DIR")
DATASET_NAME = "gtsdb"
DOWNLOAD_LINK = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip"
DOWNLOAD_PATH = os.path.join(BASE_DIR, "datasets", DATASET_NAME)


def fetchDataset(downloadLink = DOWNLOAD_LINK, downloadPath = DOWNLOAD_PATH):

	os.makedirs(downloadPath, exist_ok = True)
	zipPath = os.path.join(downloadPath, "FullIJCNN2013.zip")
	urllib.request.urlretrieve(downloadLink, zipPath)
	
	with zipfile.ZipFile(zipPath, 'r') as zipRef:
		zipRef.extractall(downloadPath)


if __name__ == "__main__":

	fetchDataset()
