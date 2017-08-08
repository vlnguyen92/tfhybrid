DATA_DIR=/tmp/data/flowers
TRAIN_DIR=/tmp/train_logs

cmd=$1

echo $cmd

nvprof --aggregate-mode on \
    ${cmd} \
    python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATA_DIR} \
    --batch_size=128 \
    --max_number_of_steps=10 \
    --model_name=alexnet_v2
