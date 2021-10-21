export NGPUS=1
sh dir.sh
CUDA_VISIBLE_DEVICES=1 python tools/test_net.py --config-file "configs/retrival_finetune2.yaml" --ckpt "model_rec_synth_ic17_7709.pth"

