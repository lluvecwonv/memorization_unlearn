master_port=29500

# forget10
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml split=forget10 npo_coeff=0.125 beta=4.5