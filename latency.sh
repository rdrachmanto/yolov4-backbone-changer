#!/bin/bash

backbone=(
        "cspdarknet53"
        "mobilenetv2_half"
)

for bb in "${backbone[@]}"
do
        for i in {1..20}
        do
                args=(
                        --backbone "${bb}"
                        --model-path "logs/yolov4_${bb}_[mask]pretrained_seed_11/best_epoch_weights.pth"
                        --classes-path "model_data/mask_classes.txt"
                        --anchors-path "model_data/mask_anchors.txt"
                        --annot-path "mask_2007_val.txt"
                )
                python3 latency.py "${args[@]}" >> hw-log-${bb}.txt
        done
done
