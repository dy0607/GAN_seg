# python3.7
"""Configuration for BaseGAN training demo.

All settings are particularly used for one replica (GPU), such as `batch_size`
and `num_workers`.
"""

runner_type = 'SegGANRunner'
gan_type = 'stylegan'
resolution = 256

base_bsz = 32
num_gpu = 4

bsz = batch_size = base_bsz // num_gpu
val_batch_size = batch_size * 2
total_img = 8000_000

# Training dataset is repeated at the beginning to avoid loading dataset
# repeatedly at the end of each epoch. This can save some I/O time.
data = dict(
    num_workers=2,
    repeat=500,
    train=dict(root_dir='data/ADEChallengeData2016/images/training', data_format='dir',
               resolution=resolution, mirror=0.5),
    val=dict(root_dir='data/ADEChallengeData2016/images/validation', data_format='dir',
             resolution=resolution),
)

controllers = dict(
    RunningLogger=dict(every_n_iters=10),
    ProgressScheduler=dict(
        every_n_iters=1, init_res=8, minibatch_repeats=4,
        lod_training_img=300_000, lod_transition_img=300_000,
        batch_size_schedule=dict(res4=bsz*4, res8=bsz*2, res16=bsz*2, res32=bsz*2, res64=bsz*2, res128=bsz),
    ),
    Snapshoter=dict(every_n_iters=2500, first_iter=True, num=100),
    FIDEvaluator=dict(every_n_iters=5000, first_iter=True, num=50000),
    Checkpointer=dict(every_n_iters=5000, first_iter=True),
)

modules = dict(
    discriminator=dict(
        model=dict(gan_type=gan_type, resolution=resolution),
        lr=dict(lr_type='FIXED'),
        opt=dict(opt_type='Adam', base_lr=1e-3, betas=(0.0, 0.99)),
        kwargs_train=dict(),
        kwargs_val=dict(),
    ),
    generator=dict(
        model=dict(gan_type=gan_type, resolution=resolution),
        lr=dict(lr_type='FIXED'),
        opt=dict(opt_type='Adam', base_lr=1e-3, betas=(0.0, 0.99)),
        kwargs_train=dict(w_moving_decay=0.995, style_mixing_prob=0.9,
                          trunc_psi=1.0, trunc_layers=0, randomize_noise=True),
        kwargs_val=dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False),
        g_smooth_img=10000,
    ),
    segmentator=dict(
        model=dict(
            gan_type=gan_type, 
            resolution=resolution, 
            config_path='config_seg/ade20k-hrnetv2.yaml')
    ),
    # segmentation_discriminator=dict(
    #     model=dict(
    #         gan_type=gan_type,
    #         resolution=resolution,
    #         image_channels=1)
    # )
)

loss = dict(
    type='SegGANLoss',
    freq_path='data/ADEChallengeData2016/seg_freq.pt',
    warmup_step=10000,
    d_loss_kwargs=dict(r1_gamma=10.0),
    g_loss_kwargs=dict(beta=10, alpha=0.0001),
)
