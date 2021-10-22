export NGPUS=1
CUDA_VISIBLE_DEVICES=1 python tools/test_net.py --config-file "configs/evaluation.yaml" --ckpt "model_rec_synth_ic17_7709.pth"

