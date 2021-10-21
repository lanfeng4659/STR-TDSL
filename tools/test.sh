export NGPUS=1
sh dir.sh
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival_chinese.yaml"
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival_chinese_1019.yaml"

CUDA_VISIBLE_DEVICES=1 python tools/test_net.py --config-file "configs/retrival_finetune2.yaml" --ckpt "Log/finetune_on_ic17_rec10_no_da_b64_640/model_0067500.pth"
# CUDA_VISIBLE_DEVICES=1 python tools/test_net.py --config-file "configs/retrival_finetune2.yaml" --ckpt "./Log/finetune_on_ic17_look_up/model_0075000.pth"
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival_finetune.yaml" --ckpt "./Log/finetune_ic13_15_17_boundary_aug/model_0002500.pth"
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival_finetune.yaml" --ckpt "./model_synthtext.pth"
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/SiamRPNRetrieval.yaml"
# CUDA_VISIBLE_DEVICES=1 python tools/test_net.py --config-file "configs/retrival.yaml" --ckpt "Log/retrival_e2e_add_retrieval_loss_10_colorjit_wordaug_b64_have_centerness_vot_veri/model_0072500.pth"
#  CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival.yaml"
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival_attention.yaml"
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival_syn_svt.yaml"
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/SiamRPN.yaml"
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival_finetune_usetextness.yaml"
# CUDA_VISIBLE_DEVICES=2 python tools/test_net.py --config-file "configs/retrival.yaml" --ckpt Log/retrival_e2e_add_retrieval_loss_10_colorjit_tanh_step_wordaug_b64_no_centerness/model_0042500.pth
#model_0052500 0.8365677274666583
#0.8311455674222604
#0.8293850682212545
#0.8377875586752732
#425000 0.8392700921492579
#0.8286347251487769
#0.8310515076845368
#0.8134350006751202
#0.8332905534717142
#0.8178556375465285
#0.8329851850891519
#0.822056137383864
#0.8115590785418934
#0.8138884682545106
#0.8294062223861857
#0.8322809870268604
#0.8335392255358354
#0.8281725410484201
#0.8166024570080435
#0.8210033786228966
#0.813244682633114
#step 0 0.818348671440315

#CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival.yaml" --ckpt "./Log/retrival_e2e_add_retrieval_loss_10/model_final.pth"

# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival.yaml" --ckpt "./Log/retrival_e2e_add_retrieval_loss_10_512_512/model_0087500.pth"
#final: 81.19
#87500: 0.8060546887819473
#0.8100832401982645
#0.8145430357501988
#0.816274963888367
#0.8102075517760523
#0.8110501087733659
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival.yaml" --ckpt "./Log/retrival_e2e_add_retrieval_loss_5/model_0087500.pth"
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival_finetune2.yaml" --ckpt "Log/finetune_on_ic17_domain3/model_0050000.pth"
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival.yaml" --ckpt Log/retrival_e2e_add_retrieval_loss_10_colorjit_wordaug_b64_have_centerness/model_final.pth
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival_ic13.yaml" --ckpt ./model_domain3_7078.pth

# python multigpu_train.py --gpu_list=2 --input_size=512 --batch_size_per_gpu=8 --checkpoint_path=./east_scenetext_5_resnet_v1_50_rbox_800epochs/ \
# --text_scale=512 --training_data_path=/home/ssd_datasets/scene_datasets_processed/ --geometry=RBOX --learning_rate=0.0001 --num_readers=4 \
# --pretrained_model_path=./resnet_v1_50.ckpt --class_list='5,' --max_steps=1500000 --restore true