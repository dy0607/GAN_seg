export CUDA_VISIBLE_DEVICES=4,5,6,7
export NGPU=4;
# export CUDA_VISIBLE_DEVICES=2,3
# export NGPU=2
nohup python \
    -m torch.distributed.launch --nproc_per_node=$NGPU \
    --master_port 29500 train.py config_gan/pgan_seg.py \
    --work_dir results/3/ > results/3/train.log &
