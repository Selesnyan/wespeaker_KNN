#!/bin/bash

# Run only with sufficient amount of memory, or train PLDA in a serial manner. It can consume quite a few GB (tens-hundreds) during PLDA training.

model=(
    "ResNet101-PS-emb256-fbank64-num_frms600-aug0.6-spTrue-saFalse-AAM-SGD-epoch150"

)

    # "SC-ResNet34-PF-emb256-fbank64-num_frms600-aug0.6-spTrue-saFalse-SGD-epoch150"
#    "ResNet101-PF-emb256-fbank64-num_frms600-aug0.6-spTrue-saFalse-AAM-SGD-epoch150"
model_name=(

   "ResNet101__"

)
#   "ResNetPerFrame101"
#   "ResNet101"
 
count=1
start_stage=3  #3 # 1 - embedding extr, 2 - cts+vox creation, 3 - plda training + dvbx
stop_stage=42

for i in $(seq 0 $(($count-1))); do
    echo ${model[$i]}
    ./extr_embeds_train_plda.sh --model_name ${model_name[$i]} \
        --exp_dir $(pwd)/exp/${model[$i]} \
        --start_stage $start_stage \
        --stop_stage $stop_stage \
        --train_plda false \
        --run_dvbx true & 
    sleep 2
done;
