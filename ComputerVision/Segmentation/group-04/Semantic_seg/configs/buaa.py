_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/cityscapes.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# data_preprocessor = dict(
#     type='SegDataPreProcessor',
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     bgr_to_rgb=True,
#     pad_val=0,
#     seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

#######################################################################################

dataset_type = 'DeepCrackDataset'
data_root = '/buaa/DeepCrack/' # 根目录
metainfo = dict(
    classes=('background', 'crack'),
    palette=[[0, 0, 0], [255, 255, 255]]
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # LoadAnnotations 会根据 `backend_args` 和 `importer` 自动处理 json
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(544, 384), keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(544, 384), keep_ratio=False),
    # a simple pipeline to test whole image
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# 3. 配置 Dataloader
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    # persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train_img/', 
            seg_map_path='train_lab/' #
        ),
        metainfo=metainfo,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    # persistent_workers=True,
    # sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test_img/',
            seg_map_path='test_lab/'
        ),
        # importer=dict(type='mmdet.CocoAnnFileParser'), 

        metainfo=metainfo,
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader


# 4. 配置 Evaluator
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU']) # 在 MMDET 3.x 风格下，用 mIoUMask 更明确
test_evaluator = val_evaluator

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'),
              dict(type='TensorboardVisBackend'),
            #   dict(type='WandbVisBackend')
              ]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')

# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=8000,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=8000, val_interval=800)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=800),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

