#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2022 Zhengyang Chen (chenzhengyang117@gmail.com)

. /mnt/matylda4/xpalka07/wespeaker/exp/vbx_16k/v2/path.sh || exit 1
#. /mnt/scratch/tmp/xkleme15/wespeaker/examples/vbx_16k/v2/path.sh || exit 1
num_gpus=4  

stage=3
stop_stage=3

data=/mnt/scratch/tmp/xpalka07/SC_DATASET/--no-use-rirs--no-use-noises/train/data
#/mnt/scratch/tmp/xpalka07/pfee/data
data_type="raw"  # shard/raw

config=/mnt/matylda4/xpalka07/wespeaker/exp/vbx_16k/v2/conf/sc_rn34pf_600frms_sgd_softmax.yaml
#conf/sc_rn34pf_600frms_sgd_softmax.yaml
#/mnt/matylda4/xpalka07/wespeaker/exp/vbx_16k/v2/conf/sc_rn34pf_600frms_sgd_BCEwithLogitsLoss.yaml 
exp_dir=/mnt/matylda4/xpalka07/wespeaker/exp/vbx_16k/v2/exp/C-SC-ResNet34-PF-emb256-fbank64-num_frms600-aug0.6-spTrue-saFalse-SGD-epoch150-softmax

#gpus=(0,1,2,3)
num_avg=10
checkpoint=$exp_dir/models/model_65.pt
#41

trials="CNC-Eval-Concat.lst CNC-Eval-Avg.lst"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning
lm_config=conf/resnet_lm.yaml

. /mnt/matylda4/xpalka07/wespeaker/exp/vbx_16k/v2/tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Preparing datasets ..."
  ./local/prepare_data.sh --stage 5 --stop_stage 5 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in cnceleb_train eval; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1024 \
          --num_threads 16 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  # Convert all musan data to LMDB
  python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  echo ${data} "${data_type}"
#  num_gpus=4 #4 #1 
  gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
#   gpus="[0,1]"
  echo "$gpus" $num_gpus 
#  exit
  torchrun --rdzv_backend=c10d --rdzv_endpoint=$(hostname):$[$RANDOM % 10000 + 20000] --nnodes=1 --nproc_per_node=$num_gpus \
    /mnt/matylda4/xpalka07/wespeaker/exp/vbx_16k/v2/wespeaker/bin/train_sc.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus "$gpus" \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/wav.scp \
      --train_label ${data}/rttm \
      --reverb_data "/mnt/scratch/tmp/xkleme15/wespeaker/examples/vbx_16k/v2/data/rirs_16k/lmdb" \
      --noise_data "/mnt/scratch/tmp/xkleme15/wespeaker/examples/vbx_16k/v2/data/musan_16k/lmdb" \
      ${checkpoint:+--checkpoint $checkpoint}
fi
#--train_label ${data}/utt2spk \

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  model_path=$avg_model
  if [[ $config == *repvgg*.yaml ]]; then
    echo "convert repvgg model ..."
    python wespeaker/models/convert_repvgg.py \
      --config $exp_dir/config.yaml \
      --load $avg_model \
      --save $exp_dir/models/convert_model.pt
    model_path=$exp_dir/models/convert_model.pt
  fi

  echo "Extract embeddings ..."
  local/extract_cnc.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 4 --gpus $gpus --data_type $data_type --data ${data}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --exp_dir $exp_dir \
    --data ${data} \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set cnceleb_train \
    --top_n $top_n \
    --exp_dir $exp_dir \
    --data ${data} \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi

# ================== Large margin fine-tuning ==================
# for reference: https://arxiv.org/abs/2206.11699
# It shoule be noted that the large margin fine-tuning
# is optional. It often be used in speaker verification
# challenge to further improve performance. This training
# proces will take longer segment as input and will take
# up more gpu memory.

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Large margin fine-tuning ..."
  lm_exp_dir=${exp_dir}-LM
  mkdir -p ${lm_exp_dir}/models
  # Use the pre-trained average model to initialize the LM training
  cp ${exp_dir}/models/avg_model.pt ${lm_exp_dir}/models/model_0.pt
  bash run.sh --stage 3 --stop_stage 7 \
      --data ${data} \
      --data_type ${data_type} \
      --config ${lm_config} \
      --exp_dir ${lm_exp_dir} \
      --gpus $gpus \
      --num_avg 1 \
      --checkpoint ${lm_exp_dir}/models/model_0.pt \
      --trials "$trials" \
      --score_norm_method ${score_norm_method} \
      --top_n ${top_n}
fi
