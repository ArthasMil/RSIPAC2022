norm_cfg = dict(type='SyncBN', requires_grad=True)
custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth',
            prefix='backbone.')),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=18,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=1.0,),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=2.0,
        ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=18,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=0.4,),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=0.8,)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.88),
    dict(
        type='Albu',
        transforms=[
            dict(type='RandomRotate90', p=0.5),
            dict(type='Transpose', p=0.3),
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.5),
            dict(type='RandomBrightnessContrast', p=0.2),
            dict(type='GaussNoise', p=0.2),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
            dict(type='RandomGridShuffle', grid=(12, 12), p=0.4),
            dict(
                type='RGBShift',
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=0.2),
            dict(type='ColorJitter', p=0.45)
        ]),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.75, 1.0, 1.25, 1.5],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=4,
    train=dict(
        type='RSIPAC',
        data_root='fusai_release',
        img_dir='train/images',
        ann_dir='train/labels',
        split='train/splits/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(
                type='Albu',
                transforms=[
                    dict(type='RandomRotate90', p=0.5),
                    dict(type='Transpose', p=0.3),
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.2,
                        scale_limit=0.2,
                        rotate_limit=30,
                        p=0.5),
                    dict(type='RandomBrightnessContrast', p=0.2),
                    dict(type='GaussNoise', p=0.2),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.1),
                    dict(type='RandomGridShuffle', grid=(12, 12), p=0.4),
                    dict(
                        type='RGBShift',
                        r_shift_limit=20,
                        g_shift_limit=20,
                        b_shift_limit=20,
                        p=0.2),
                    dict(type='ColorJitter', p=0.45)
                ]),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='RSIPAC',
        data_root='fusai_release',
        img_dir='train/images',
        ann_dir='train/labels',
        split='train/splits/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='RSIPAC',
        data_root='fusai_release',
        img_dir='test/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                img_ratios=[0.75, 1.0, 1.25, 1.5],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=True)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=80)
checkpoint_config = dict(by_epoch=True, interval=10)
evaluation = dict(
    by_epoch=True, interval=10, metric=['mIoU', 'mFscore'], pre_eval=True)
fp16 = dict()
work_dir = 'myconfigs/conv-b-512/'
gpu_ids = [0]
auto_resume = False
