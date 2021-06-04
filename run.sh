export CUDA_VISIBLE_DEVICES=0,1,2,5
export NGPU=4
# export CUDA_VISIBLE_DEVICES=2,3
# export NGPU=2
nohup python \
    -m torch.distributed.launch --nproc_per_node=$NGPU \
    --master_port 29500 train.py config_gan/pgan_seg.py \
    --work_dir results/ade_loss0/ > results/ade_loss0/train.log &