import os
import urllib.request
import zipfile


BASE_DIR = os.environ.get("BASE_DIR")
DATASET_NAME = "gtsrb"
DOWNLOAD_LINK_TRAIN = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
DOWNLOAD_LINK_TEST = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
DOWNLOAD_LINKS = [DOWNLOAD_LINK_TRAIN, DOWNLOAD_LINK_TEST]
DOWNLOAD_PATH = os.path.join(BASE_DIR, "datasets", DATASET_NAME)


def fetchDataset(downloadLink = DOWNLOAD_LINKS, downloadPath = DOWNLOAD_PATH):

	os.makedirs(downloadPath, exist_ok = True)
	zipPathTrain = os.path.join(downloadPath, "GTSRB_Final_Training_Images.zip")
	zipPathTest = os.path.join(downloadPath, "GTSRB_Final_Test_Images.zip")
#	urllib.request.urlretrieve(downloadLink[0], zipPathTrain)
#	urllib.request.urlretrieve(downloadLink[1], zipPathTest)
	
#	with zipfile.ZipFile(zipPathTrain, 'r') as zipRef:
#		zipRef.extractall(downloadPath)

	with zipfile.ZipFile(zipPathTest, 'r') as zipRef:
		zipRef.extractall(downloadPath)

if __name__ == "__main__":

	fetchDataset()
