#!/bin/bash

# Copyright 2023 Dominik Klement (xkleme15@vutbr.cz)
# The workflow is as follows:
#   1. Take an arbitrary dataset with its vad, wav.scp, utt2spk files
#   2. Create raw.list using tools/make_raw_list.py
#   3. Create augmented raw.list, wav.scp, utt2spk, spk2utt - using tools/create_aug_repl_dset.py and perl utt2spk_to_spk2utt.pl
#   4. Enjoy the features


#. /mnt/scratch/tmp/xkleme15/wespeaker/examples/vbx_16k/v2/path.sh || exit 1
. /mnt/matylda4/xpalka07/wespeaker/exp/vbx_16k/v2/path.sh || exit 1

stage=10
stop_stage=10

# data=/mnt/ssd/ws_data/data
data=data
data_type="shard"  # shard/raw
aug_plda_data=0

config=
#/mnt/scratch/tmp/xkleme15/wespeaker/examples/vbx_16k/v2/conf/rn101pf_600frms_sgd_aam.yaml
# rn34pf_400frms_sgd_aam.yaml
exp_dir=
#/mnt/matylda4/xpalka07/wespeaker/examples/vbx_16k/v2/exp/ResNet101-PS2PF-emb256-fbank64-num_frms600-aug0.6-spTrue-saFalse-AAM-SGD-epoch150
#rn101pf_test
#ResNet34-PF-emb256-fbank64-num_frms400-aug0.6-spTrue-saFalse-AAM-SGD-epoch150
gpus="[0,1]"
num_avg=10
checkpoint=

trials="CNC-Eval-Concat.lst CNC-Eval-Avg.lst"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning
lm_config=conf/resnet_lm.yaml

use_rand_chunk=true
rand_chunk_len=600

. /mnt/matylda4/xpalka07/wespeaker/exp/vbx_16k/v2/tools/parse_options.sh || exit 1

# Feature extraction.
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  # It's more like a number of processes.
  gpus="[0,1,2,3]"
  num_gpus=1
  avg_model=$exp_dir/models/avg_model.pt
  model_path=$avg_model

  echo "Extract Features ..."
  local/extract_feats.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj $num_gpus --gpus $gpus --data_type raw --data ${data} \
    --reverb_data ${data}/rirs_16k/lmdb \
    --noise_data ${data}/musan_16k/lmdb \
    --save-batch-size 1024 \
    --use_rand_chunk $use_rand_chunk \
    --rand_chunk_len $rand_chunk_len \
    --aug_plda_data ${aug_plda_data}

    /mnt/matylda4/landini/scripts/manage_task.sh -q short.q@@blade -l ram_free=8G,mem_free=8G $exp_dir/feats_rc_600/vox_cn1_aug/fea_extr_task
fi
