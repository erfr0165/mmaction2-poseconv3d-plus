_base_ = '../../_base_/default_runtime.py'

dimensions = 20
embedding_path = '/data/embeddings/ikea_20dimensions.pkl'

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=dimensions,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=33,
        dropout_ratio=0.5,
        average_clips='prob'))

dataset_type = 'PoseDataset'
ann_file_train = '/data/ikea/train_no_object_all_devs.pkl'
ann_file_val = '/data/ikea/test_no_object_all_devs.pkl'
left_kp = []  # empty, since the heatmaps are not responsible for single keypoints 
right_kp = []
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=54),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTargetWithEmbeddings',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        dimensions=dimensions,
        embedding_path=embedding_path),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=54, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTargetWithEmbeddings',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        dimensions=dimensions,
        embedding_path=embedding_path),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='UniformSampleFrames', clip_len=54, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTargetWithEmbeddings',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        left_kp=left_kp,
        right_kp=right_kp,
        dimensions=dimensions,
        embedding_path=embedding_path),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            split='train',
            pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        split='test',
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        split='test',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=240, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=240,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0003),
    clip_grad=dict(max_norm=40, norm_type=2))

log_processor = dict(type='LogProcessor', window_size=100, by_epoch=True)
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1000, ignore_last=False, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='acc/mean1', max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))