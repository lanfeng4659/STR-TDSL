# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival_ic13.yaml" --ckpt ./model_domain3_7078.pth
CUDA_VISIBLE_DEVICES=2 python tools/test_net.py --config-file "configs/retrival_cocotext_anno.yaml" --ckpt ./model_rec_7052.pth
# python tools/show_ic13.py
# python evaluation/Pascal_VOC.py 