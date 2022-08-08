_base_ = [
    './swin/swin_large.py', './default_runtime.py'
]

# load_from='/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/weight/swin_base_patch244_window877_kinetics400_22k.pth'
# load_from='/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/weight/swin_base_patch244_window877_kinetics600_22k.pth'
# load_from='/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/result/v2-pretrained/epoch_80.pth'
# load_from='/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/result/v5-sigmoid/v2/epoch_100.pth'
# load_from='/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/result/v5-sigmoid-momentum-score/v1/epoch_26.pth'
# load_from='/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/swin_large_patch4_window12_384_22k.pth'
load_from = '/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/result/v6-swin-large/v2/epoch_100.pth'
model=dict(
    backbone=dict(
        # pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth',
        # pretrained='/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/weight/swin_base_patch244_window877_kinetics600_22k.pth',
        # pretrained='/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/weight/swin_base_patch244_window877_kinetics400_22k.pth',
        # pretrained='/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/weight/swin_large_patch4_window12_384_22k.pth',
        patch_size=(2,4,4), drop_path_rate=0.3
    ), 
    # train_cfg=dict(
    #     blending=dict(type='MixupBlending', num_classes=1, alpha=.2)),
    test_cfg=dict(max_testing_views=4)
)
# load_from='/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/result/test/epoch_30.pth'

# dataset settings
# dataset_type = 'VideoDataset'
dataset_type = 'UrbanPipe'
data_root = 'data/urbanpipe_data/media/sdd/zhangxuan/eccv_data_raw_video'
data_root_val = 'data/urbanpipe_data/media/sdd/zhangxuan/eccv_data_raw_video'
ann_file_train = 'data/urbanpipe/train.json'
# ann_file_test = 'data/urbanpipe/test.json'
ann_file_test = 'data/urbanpipe/development.json'
ann_file_val = ann_file_train
# ann_file_test = ann_file_train
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    # dict(type='SampleFrames', clip_len=160, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        # clip_len=128,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        # clip_len=160,
        frame_interval=2,
        num_clips=4,
        test_mode=True),
    dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 224)),
    # dict(type='ThreeCrop', crop_size=224),
    dict(type='Resize', scale=(-1, 384)),
    dict(type='ThreeCrop', crop_size=384),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=['frame_dir']),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    # videos_per_gpu=2,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 100

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/k400_swin_base_patch244_window877.py'
find_unused_parameters = False


# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=8,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    # use_fp16=True,
    use_fp16=False,
)
