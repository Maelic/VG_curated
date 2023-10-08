#!/bin/bash
set -x
set -e

SPLIT_NAME=VG80K
OUT_PATH=./outputs/$SPLIT_NAME

N_OBJ=179 # number of object categories
N_REL=49 # number of relationship categories

H5=VG-SGG.h5
JSON=VG-SGG-dicts.json
IMDB=inputs/imdb_1024.h5 # path to imdb.h5 file
OBJECTS=inputs/object_list.txt
PREDICATES=inputs/predicate_list.txt
IMAGES_LIST=inputs/indoor_vg.csv # for custom dataset
PART_WHOLE=inputs/VG80K_filtered_conceptnet_haspart_cosine.csv
OBJ_ALIAS=inputs/object_alias_without_gender.txt
PRED_ALIAS=inputs/predicate_alias_curated.txt

python main.py \
    --imdb $IMDB \
    --json_file $OUT_PATH/$JSON \
    --h5_file $OUT_PATH/$H5 \
    --object_input inputs/objects.json \
    --relationship_input inputs/relationships.json  \
    --num_objects $N_OBJ \
    --num_predicates $N_REL \
    --save_bboxes True \
    --yolo_anno False \
    --object_alias $OBJ_ALIAS \
    --pred_alias $PRED_ALIAS \
    --metadata_input inputs/image_data.json \
    --use_negative_lists True