
export NGPUS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20008 tools/train_net.py --config-file "configs/retrival_finetune2.yaml"

