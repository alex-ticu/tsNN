
PYTHON = python3

BASE_DIR = $(shell pwd)

gtsdb: 
	export BASE_DIR=${BASE_DIR}; \
	${PYTHON} ./datasets/getGTSDetectionBenchmark.py
