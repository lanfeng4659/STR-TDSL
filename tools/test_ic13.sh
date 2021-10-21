# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival_ic13.yaml" --ckpt ./model_domain3_7078.pth
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/retrival_ic13.yaml" --ckpt ./model_domain3_7674.pth
python tools/show_ic13.py
python evaluation/Pascal_VOC.py 