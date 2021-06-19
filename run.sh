export CUDA_VISIBLE_DEVICES=4,5,6,7
export NGPU=4
# export CUDA_VISIBLE_DEVICES=2,3
# export NGPU=2
mkdir results/ade_seg4
nohup python -u\
    -m torch.distributed.launch --nproc_per_node=$NGPU \
    --master_port 29503 train.py config_gan/pgan_seg4.py \
    --work_dir results/ade_seg4/ > results/ade_seg4/train.log &



# nohup python -u\
#     -m torch.distributed.launch --nproc_per_node=$NGPU \
#     --master_port 29502 train.py config_gan/pgan_seg3.py \
#     --work_dir results/ade_seg3/ > results/ade_seg3/train.log &

:<<!
nohup python -u\
    -m torch.distributed.launch --nproc_per_node=$NGPU \
    --master_port 29500 train.py config_gan/pgan_seg.py \
    --work_dir results/ade_loss2/ > results/ade_loss2/train.log &
!

:<<!
nohup python -u\
    -m torch.distributed.launch --nproc_per_node=$NGPU \
    --master_port 29503 train.py config_gan/pgan_ade.py \
    --work_dir results/ade_base1/ > results/ade_base1/train.log --resume_path results/ade_base0/checkpoint_iter140000.pth &
!