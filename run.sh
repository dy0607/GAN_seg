export CUDA_VISIBLE_DEVICES=4,5,6,7
export NGPU=4;
nohup python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port 29501 train.py config_gan/pgan_ade.py  --work_dir results/1/ > results/train.log &