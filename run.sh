export CUDA_VISIBLE_DEVICES=4,5,6,7
export NGPU=4
# export CUDA_VISIBLE_DEVICES=2,3
# export NGPU=2
mkdir results/ade_loss_1
nohup python -u\
    -m torch.distributed.launch --nproc_per_node=$NGPU \
    --master_port 29501 train.py config_gan/pgan_seg2.py \
    --work_dir results/ade_loss_1/ > results/ade_loss_1/train.log &
