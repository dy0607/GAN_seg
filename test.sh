export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=4
# export CUDA_VISIBLE_DEVICES=2,3
# export NGPU=2
mkdir results/val_loss_1
python -m torch.distributed.launch --nproc_per_node=$NGPU \
    --master_port 29501 test.py config_gan/seg_val.py \
    --work_dir results/val_loss_1 --checkpoint "results/ade_loss_1/checkpoint_iter217184.pth" --config_path "config_seg/ade20k-hrnetv2.yaml"
