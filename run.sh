export CUDA_VISIBLE_DEVICES=3
export NGPU=1;
# export CUDA_VISIBLE_DEVICES=2,3
# export NGPU=2
nohup python -u\
    -m torch.distributed.launch --nproc_per_node=$NGPU \
    --master_port 29502 train.py config_gan/pgan_seg.py \
    --resume_path /home/duanyu/dl/gan/results/3/checkpoint_iter050000.pth\
    --work_dir results/4/ > results/4/train.log &
