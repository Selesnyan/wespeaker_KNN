#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2023 Zhengyang Chen (chenzhengyang117@gmail.com)

. /mnt/matylda4/xpalka07/wespeaker/exp/vbx_16k/v2/path.sh || exit 1

# data=/mnt/ssd/ws_data/data
data=/mnt/scratch/tmp/xkleme15/wespeaker/examples/vbx_16k/v2/data
data_type="shard"  # shard/raw
# whether augment the PLDA data
aug_plda_data=0

exp_dir="none"
num_avg=10
checkpoint=""

trials="trials trials_tgl trials_yue"

USE_CUDA=1

start_stage=-1
stop_stage=42
model_name=ResNet # ResNetPerFrame101 # "ResNetPerFrame34"
train_plda=true
run_dvbx=true

# Different set may use different backend training sets, therefore we need several trial lists
# Using "," instead of space as separator is a bit ugly but it seems parse_options,sh connot
# process an argument with space properly.

. /mnt/matylda4/xpalka07/wespeaker/exp/vbx_16k/v2/tools/parse_options.sh || exit 1

echo "Start: ${start_stage}, Stop: ${stop_stage}, Model: ${model_name}, ExpDir: ${exp_dir}"
$train_plda && (echo "Training Plda")
$run_dvbx && (echo "Running DVBx")


echo "EXP_DIR: {$exp_dir}"

avg_model=$exp_dir/models/avg_model.pt
if [ ! -f $avg_model ]; then 
    echo "Do model average ..."
    python /mnt/matylda4/xpalka07/wespeaker/exp/vbx_16k/v2/wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}
fi

model_path=$avg_model
NUM_EMBEDDINGS=1

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

    # avg_model=$exp_dir/models/avg_model.pt
    # if [ ! -f $avg_model ]; then 
    #     echo "Do model average ..."
    #     python /mnt/scratch/tmp/xkleme15/wespeaker/examples/vbx_16k/v2/wespeaker/bin/average_model.py \
    #     --dst_model $avg_model \
    #     --src_path $exp_dir/models \
    #     --num ${num_avg}
    # fi

    # model_path=$avg_model
    rm $exp_dir/embed_extr_task

    echo "Extract embeddings From Features ..."
    # nj in this case is number of jobs one wants to create and submit.
    # num_embeddings specifies how many unique per-frame embeddings we want to extract from a single utterance.

    for num_embeddings in $NUM_EMBEDDINGS; do
        rm $exp_dir/embed_${num_embeddings}_extr_task

        nj=-1
        if [ $USE_CUDA -eq 1 ]; then
            nj=50
        fi

        local/extract_sre_from_feats.sh --num_embeddings "${num_embeddings}" \
        --exp_dir "$exp_dir" --model_path $model_path \
        --nj "$nj" --data_type "feats" --data "${data}" \
        --feats "${data}/pre_extr_feats_rc_600" \
        --reverb_data "${data}/rirs_16k/lmdb" \
        --noise_data "${data}/musan_16k/lmdb" \
        --use_cuda "${USE_CUDA}" \
        --aug_plda_data "${aug_plda_data}"
        #echo "dont sub"
        #exit
        q_type="all.q"
        all_gpu_queues=$(echo $(qstat -f -xml | grep $q_type | grep gpu | sed 's#<[^>]*>##g' | awk '{print $1}') | tr '[:blank:]', ',')
        # all_gpu_queues="all.q@supergpu10,all.q@supergpu11,all.q@supergpu12,all.q@supergpu13,all.q@supergpu14"
        # all_gpu_queues="all.q@@blade"

        if [ $USE_CUDA -eq 1 ]; then
            /mnt/matylda4/landini/scripts/manage_task.sh -q $all_gpu_queues -sync yes -l gpu=1,ram_free=32G,mem_free=32G $exp_dir/embed_${num_embeddings}_rc_600_feats_extr_task &> $exp_dir/embed_${num_embeddings}_extr_task_out
        else 
            /mnt/matylda4/landini/scripts/manage_task.sh -q $all_gpu_queues -sync yes -l ram_free=4G,mem_free=4G $exp_dir/embed_${num_embeddings}_rc_600_feats_extr_task &> $exp_dir/embed_${num_embeddings}_extr_task_out
        fi
    done

    for emb_path in $exp_dir/embeddings*/*; do
            
        cat ${emb_path}/xvector_*.scp > ${emb_path}/xvector.scp
        python local/xvec_scp_to_utt2spk.py --xvec_scp_path ${emb_path}/xvector.scp
        tools/utt2spk_to_spk2utt.pl ${emb_path}/utt2spk > ${emb_path}/spk2utt

    done
fi

embeddings_dir="embeddings_${NUM_EMBEDDINGS}_rc_600_feats"

# WE HAVE NO CTS 16K, THEREFORE SKIPPING!
# If $train_plda is false, we don't need to create cts_vox_aug as we don't need the datasets anymore.
# For DVBx, it was either created before or is not going to be used for training.
# if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && $train_plda; then

#     # Create cts+vox dataset
#     if [ -d ${exp_dir}/${embeddings_dir}/vox_aug ]; then
#         echo "Creating CTS+VOX"
#         data_dir=${exp_dir}/${embeddings_dir}/cts_vox_aug
#         mkdir -p $data_dir

#         cat ${exp_dir}/${embeddings_dir}/cts-bk_aug/xvector.scp > $data_dir/xvector.scp
#         cat ${exp_dir}/${embeddings_dir}/vox_aug/xvector.scp >> $data_dir/xvector.scp

#         cat ${exp_dir}/${embeddings_dir}/cts-bk_aug/utt2spk > $data_dir/utt2spk
#         cat ${exp_dir}/${embeddings_dir}/vox_aug/utt2spk >> $data_dir/utt2spk

#         cat ${exp_dir}/${embeddings_dir}/cts-bk_aug/spk2utt > $data_dir/spk2utt
#         cat ${exp_dir}/${embeddings_dir}/vox_aug/spk2utt >> $data_dir/spk2utt
#     else
#         echo "No Vox Dataset, skipping CTS+VOX creation"
#     fi

# fi


if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

    for train_data in "vox_cn1_aug"; do 
        echo $train_data "${exp_dir}/$embeddings_dir/${train_data}"
        if [ -d ${exp_dir}/$embeddings_dir/${train_data} ]; then
            if $train_plda; then
                echo "Training PLDA for ${train_data}"
            fi

            for lda in 128; do
                {
                    data_dir=${exp_dir}/$embeddings_dir/
                    lda_set=$train_data
                    # lda_u2s=data/$train_data/utt2spk
                    lda_u2s=${exp_dir}/$embeddings_dir/$train_data/utt2spk
                    plda_set=$train_data  
                    # plda_s2u=data/$train_data/spk2utt
                    plda_s2u=${exp_dir}/$embeddings_dir/$train_data/spk2utt
                    adp_set=None
                    evl_cnd=evl_cond
                    #trial_list_dir=data/sre21/eval
                    trial_list_dir=data/sre21/dev
                    lda_dim=$lda
                    plda_dir=${exp_dir}/plda__${embeddings_dir}/${train_data}_lda_${lda}
                    mkdir -p $plda_dir
                    echo "LDA dim.: $lda"

                    if $train_plda; then
                        export LD_LIBRARY_PATH="/mnt/matylda5/iplchot/python_public/anaconda3/lib/:$LD_LIBRARY_PATH";
                        ./local/evalauate_xvec_plda_kaldi.sh $data_dir $lda_set $lda_u2s $plda_set $plda_s2u $adp_set $evl_cnd $trial_list_dir $lda_dim $plda_dir
                    fi

                    if $run_dvbx; then
                        {
                            /mnt/scratch/tmp/xkleme15/DiscriminativeVBx/prepare_train_set16k_and_train_general.sh "dh_dev" "${plda_dir}/dh_data_dev_corr_timing" "$model_path" "$model_name" "$plda_dir"
                            /mnt/scratch/tmp/xkleme15/DiscriminativeVBx/prepare_train_set16k_and_train_general.sh "dh_eval" "${plda_dir}/dh_data_eval_corr_timing" "$model_path" "$model_name" "$plda_dir"
                        } &
                    fi
                } &
            done

        else
            echo "${exp_dir}/$embeddings_dir/${train_data} not found."
        fi
    done

fi


# #!/bin/bash

# # Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
# #           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
# #           2023 Zhengyang Chen (chenzhengyang117@gmail.com)

# . /mnt/scratch/tmp/xkleme15/wespeaker/examples/vbx_16k/v2/path.sh || exit 1

# # data=/mnt/ssd/ws_data/data
# data=/mnt/scratch/tmp/xkleme15/wespeaker/examples/vbx_16k/v2/data
# data_type="shard"  # shard/raw
# # whether augment the PLDA data
# aug_plda_data=0

# exp_dir="none"
# num_avg=10
# checkpoint=""

# trials="trials trials_tgl trials_yue"

# USE_CUDA=1

# start_stage=-1
# stop_stage=42
# model_name="ResNetPerFrame34"

# # Different set may use different backend training sets, therefore we need several trial lists
# # Using "," instead of space as separator is a bit ugly but it seems parse_options,sh connot
# # process an argument with space properly.

# . /mnt/scratch/tmp/xkleme15/wespeaker/examples/vbx_16k/v2/tools/parse_options.sh || exit 1

# echo "Start: ${start_stage}, Stop: ${stop_stage}, Model: ${model_name}, ExpDir: ${exp_dir}"

# avg_model=$exp_dir/models/avg_model.pt
# if [ ! -f $avg_model ]; then 
#     echo "Do model average ..."
#     python /mnt/scratch/tmp/xkleme15/wespeaker/examples/vbx_16k/v2/wespeaker/bin/average_model.py \
#     --dst_model $avg_model \
#     --src_path $exp_dir/models \
#     --num ${num_avg}
# fi

# model_path=$avg_model
# NUM_EMBEDDINGS=1

# if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

#     # avg_model=$exp_dir/models/avg_model.pt
#     # if [ ! -f $avg_model ]; then 
#     #     echo "Do model average ..."
#     #     python /mnt/scratch/tmp/xkleme15/wespeaker/examples/sre/v2/wespeaker/bin/average_model.py \
#     #     --dst_model $avg_model \
#     #     --src_path $exp_dir/models \
#     #     --num ${num_avg}
#     # fi

#     # model_path=$avg_model
#     rm $exp_dir/embed_extr_task

#     echo "Extract embeddings From Features ..."
#     # nj in this case is number of jobs one wants to create and submit.
#     # num_embeddings specifies how many unique per-frame embeddings we want to extract from a single utterance.

#     for num_embeddings in $NUM_EMBEDDINGS; do
#         rm $exp_dir/embed_${num_embeddings}_extr_task

#         nj=-1
#         if [ $USE_CUDA -eq 1 ]; then
#             nj=50
#         fi

#         local/extract_sre_from_feats.sh --num_embeddings ${num_embeddings} \
#         --exp_dir $exp_dir --model_path $model_path \
#         --nj $nj --data_type feats --data ${data} \
#         --feats ${data}/pre_extr_feats \
#         --reverb_data ${data}/rirs_16k/lmdb \
#         --noise_data ${data}/musan_16k/lmdb \
#         --use_cuda ${USE_CUDA} \
#         --aug_plda_data ${aug_plda_data}

#         q_type="all.q"
#         all_gpu_queues=$(echo $(qstat -f -xml | grep $q_type | grep gpu | sed 's#<[^>]*>##g' | awk '{print $1}') | tr '[:blank:]', ',')
#         # all_gpu_queues="all.q@supergpu10,all.q@supergpu11,all.q@supergpu12,all.q@supergpu13,all.q@supergpu14"
#         # all_gpu_queues="all.q@@blade"

#         if [ $USE_CUDA -eq 1 ]; then
#             /mnt/matylda4/landini/scripts/manage_task.sh -q $all_gpu_queues -sync yes -l gpu=1,ram_free=32G,mem_free=32G $exp_dir/embed_${num_embeddings}_extr_task
#         else 
#             /mnt/matylda4/landini/scripts/manage_task.sh -q $all_gpu_queues -sync yes -l ram_free=4G,mem_free=4G $exp_dir/embed_${num_embeddings}_extr_task
#         fi
#     done

#     for emb_path in $exp_dir/embeddings*/*; do
            
#         cat ${emb_path}/xvector_*.scp > ${emb_path}/xvector.scp
#         python local/xvec_scp_to_utt2spk.py --xvec_scp_path ${emb_path}/xvector.scp
#         tools/utt2spk_to_spk2utt.pl ${emb_path}/utt2spk > ${emb_path}/spk2utt

#     done
# fi

# embeddings_dir="embeddings_${NUM_EMBEDDINGS}"

# # if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

# #     # Create cts+vox dataset
# #     if [ -d ${exp_dir}/${embeddings_dir}/vox_aug ]; then
# #         echo "Creating CTS+VOX"
# #         data_dir=${exp_dir}/${embeddings_dir}/cts_vox_aug
# #         mkdir -p $data_dir

# #         cat ${exp_dir}/${embeddings_dir}/cts-bk_aug/xvector.scp > $data_dir/xvector.scp
# #         cat ${exp_dir}/${embeddings_dir}/vox_aug/xvector.scp >> $data_dir/xvector.scp

# #         cat ${exp_dir}/${embeddings_dir}/cts-bk_aug/utt2spk > $data_dir/utt2spk
# #         cat ${exp_dir}/${embeddings_dir}/vox_aug/utt2spk >> $data_dir/utt2spk

# #         cat ${exp_dir}/${embeddings_dir}/cts-bk_aug/spk2utt > $data_dir/spk2utt
# #         cat ${exp_dir}/${embeddings_dir}/vox_aug/spk2utt >> $data_dir/spk2utt
# #     else
# #         echo "No Vox Dataset, skipping CTS+VOX creation"
# #     fi

# # fi

# if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

#     for train_data in "vox_cn1_aug"; do 
#         echo $train_data "${exp_dir}/$embeddings_dir/${train_data}"
#         if [ -d ${exp_dir}/$embeddings_dir/${train_data} ]; then
#             echo "Training PLDA for ${train_data}"

#             for lda in 128; do
#                 {
#                     data_dir=${exp_dir}/$embeddings_dir/
#                     lda_set=$train_data
#                     # lda_u2s=data/$train_data/utt2spk
#                     lda_u2s=${exp_dir}/$embeddings_dir/$train_data/utt2spk
#                     plda_set=$train_data  
#                     # plda_s2u=data/$train_data/spk2utt
#                     plda_s2u=${exp_dir}/$embeddings_dir/$train_data/spk2utt
#                     adp_set=None
#                     evl_cnd=evl_cond
#                     #trial_list_dir=data/sre21/eval
#                     trial_list_dir=data/sre21/dev
#                     lda_dim=$lda
#                     plda_dir=${exp_dir}/plda__${embeddings_dir}/${train_data}_lda_${lda}
#                     mkdir -p $plda_dir
#                     echo "LDA dim.: $lda"
#                     export LD_LIBRARY_PATH="/mnt/matylda5/iplchot/python_public/anaconda3/lib/:$LD_LIBRARY_PATH";
#                     ./local/evalauate_xvec_plda_kaldi.sh $data_dir $lda_set $lda_u2s $plda_set $plda_s2u $adp_set $evl_cnd $trial_list_dir $lda_dim $plda_dir

#                     {
#                         /mnt/scratch/tmp/xkleme15/DiscriminativeVBx/prepare_train_set16k_and_train_general.sh "dh_dev" "${plda_dir}/dh_data_dev" "$model_path" "$model_name" "$plda_dir"
#                         /mnt/scratch/tmp/xkleme15/DiscriminativeVBx/prepare_train_set16k_and_train_general.sh "dh_eval" "${plda_dir}/dh_data_eval" "$model_path" "$model_name" "$plda_dir"
#                     } &
#                 }
#             done

#         else
#             echo "${exp_dir}/$embeddings_dir/${train_data} not found."
#         fi
#     done

# fi
