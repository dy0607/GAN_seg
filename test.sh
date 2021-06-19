export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=4
# export CUDA_VISIBLE_DEVICES=2,3
# export NGPU=2
# mkdir results/val_loss_2
# python -m torch.distributed.launch --nproc_per_node=$NGPU \
#     --master_port 29501 test.py config_gan/seg_val.py \
#     --work_dir results/val_loss_2 --checkpoint "results/ade_loss2/checkpoint_iter217184.pth" --config_path "config_seg/ade20k-hrnetv2.yaml"

mkdir results/val_seg_3
python -m torch.distributed.launch --nproc_per_node=$NGPU \
    --master_port 29499 test.py config_gan/seg_val.py \
    --work_dir results/val_seg_3 --checkpoint "results/ade_seg3/checkpoint_iter275000.pth" --config_path "config_seg/ade20k-hrnetv2.yaml"

