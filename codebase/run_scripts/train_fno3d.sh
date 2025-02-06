#!/bin/bash

DROPBOX="/home/michael/GaTech Dropbox/Michael Probst/Shared/SML_project"

echo $DROPBOX

# CONDAENV=$DROPBOX/conda
CONDAENV=/home/michael/Github/comptics/comptics
DATA="$DROPBOX"
LOGS="$DROPBOX"/output/logs
CKPTS="$DROPBOX"/output/checkpoints/

# DATA=$(ws_find fno)/datasets/fields_3d_exeyez/
# CKPTS=$(ws_find fno)/checkpoints/

mkdir -p "$LOGS"
mkdir -p "$CKPTS"


TRAIN=128
VAL=32
BATCH=4
EPOCHS=100

mpirun -np 2 python codebase/train_surrogates.py \
    --name fno3d_128 \
    --model fno3d \
    --modes 12 --width 32 --blocks 10 --padding 2 \
    --split $TRAIN $VAL --batch-size $BATCH --epochs $EPOCHS \
    --data-key design --label-key fields \
    --lr 1e-3 --weight-decay 1e-6 --scheduler onecycle \
    --accelerator gpu --devices 4 --strategy ddp_find_unused_parameters_false \
    --num-workers 16 \
    --data-dir "$DATA" --log-dir "$LOGS" --checkpoint-dir "$CKPTS"
