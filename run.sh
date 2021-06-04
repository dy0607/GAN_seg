export CUDA_VISIBLE_DEVICES=1,2,3,4
export NGPU=4
# export CUDA_VISIBLE_DEVICES=2,3
# export NGPU=2
nohup python -u\
    -m torch.distributed.launch --nproc_per_node=$NGPU \
    --master_port 29501 train.py config_gan/pgan_ade.py \
    --work_dir results/ade_base0/ > results/ade_base0/train.log &
