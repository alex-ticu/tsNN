import os
import urllib.request
import zipfile


BASE_DIR = os.environ.get("BASE_DIR")
DATASET_NAME = "gtsrb"
DOWNLOAD_LINK_TRAIN = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
DOWNLOAD_LINK_TEST = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
DOWNLOAD_LINKS = [DOWNLOAD_LINK_TRAIN, DOWNLOAD_LINK_TEST]
DOWNLOAD_PATH = os.path.join(BASE_DIR, "datasets", DATASET_NAME)

translatedLabels = [ \
		"SpeedLimit_20",
		"SpeedLimit_30",
		"SpeedLimit_50",
		"SpeedLimit_60",
		"SpeedLimit_70",
		"SpeedLimit_80",
		"EndRestriction_SpeedLimit_80",
		"SpeedLimit_100",
		"SpeedLimit_120",
		"OvertakingProhibited",
		"OvertakingProhibited_Trucks",
		"Crossroad_SideRoads",
		"BeginPriorityRoad",
		"GiveWay",
		"Stop",
		"EntryProhibited",
		"EntryProhibited_Trucks",
		"EntryProhibited_OneWayTraffic",
		"OtherDangers",
		"CurveLeft",
		"CurveRight",
		"CurveSuccesion_FirstLeft",
		"BumpyRoad",
		"SlipperyRoad",
		"RoadNarrowing_Right",
		"RoadWorks",
		"TrafficLight",
		"Pedestrians",
		"School",
		"Bycicles",
		"FrostWarning",
		"DeerWarning",
		"EndOfAllRestrictions",
		"TurnRight",
		"TurnLeft",
		"Go_Straight",
		"Go_Straight_Or_TurnRight",
		"GoS_traight_Or_TurnLeft",
		"Passing_right",
		"Passing_left",
		"Roundabout",
		"OvertakingAllowed",
		"OvertakingAllowed_Trucks"
	]


def fetchDataset(downloadLink = DOWNLOAD_LINKS, downloadPath = DOWNLOAD_PATH):

	os.makedirs(downloadPath, exist_ok = True)
	zipPathTrain = os.path.join(downloadPath, "GTSRB_Final_Training_Images.zip")
	zipPathTest = os.path.join(downloadPath, "GTSRB_Final_Test_Images.zip")
	urllib.request.urlretrieve(downloadLink[0], zipPathTrain)
	urllib.request.urlretrieve(downloadLink[1], zipPathTest)
	
	with zipfile.ZipFile(zipPathTrain, 'r') as zipRef:
		zipRef.extractall(downloadPath)

	with zipfile.ZipFile(zipPathTest, 'r') as zipRef:
		zipRef.extractall(downloadPath)


def translateLabels(datasetPath = DOWNLOAD_PATH):

	labelsName = "labels.txt"
	labelsPath = os.path.join(DOWNLOAD_PATH, labelsName)

	if os.path.exists(labelsPath):
		ans = input("%s already exists. Overwrite it? (y/n)" % labelsPath)
		if ans == "n":
			return

	with open(labelsPath, "w") as fp:
		
		for label in translatedLabels:
			fp.write(label + "\n")


if __name__ == "__main__":

	fetchDataset()
	translateLabels()
