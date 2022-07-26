#!/bin/bash

DATA_DIR="../../../data/${DATASET}"
ANNO_DIR="../../../data/${DATASET}/annotations"
python download.py ${ANNO_DIR}/kinetics_train.csv ${DATA_DIR}/videos_train
python download.py ${ANNO_DIR}/kinetics_val.csv ${DATA_DIR}/videos_val
