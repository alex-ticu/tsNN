include ./common/generalMakefile.mk

PYTHON := python3

BASE_DIR = $(shell pwd)

gtsdb: 
	export BASE_DIR=${BASE_DIR}; \
	${PYTHON} ./datasets/getGTSDetectionBenchmark.py

gtsrb:
	export BASE_DIR=${BASE_DIR}; \
	${PYTHON} ./datasets/getGTSRecognitionBenchmark.py

test:
	echo ${COMMON_PATH}
