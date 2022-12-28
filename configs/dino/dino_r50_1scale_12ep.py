_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

num_feature_levels = 4
num_queries = 900
num_classes = 80

detector = dict(
    type='DINO',
    backbone=dict(
        type='D2ResNet',
        stem=dict(in_channels=3, out_channels=64, norm='FrozenBN'),
        stages=dict(depth=50, stride_in_1x1=False, norm='FrozenBN'),
        out_features=['res3', 'res4', 'res5'],
        freeze_at=1),
    position_embedding=dict(
        type='PositionEmbeddingSine',
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5),
    neck=dict(
        type='ChannelMapper',
        input_shapes={
            'res3': dict(channels=512),
            'res4': dict(channels=1024),
            'res5': dict(channels=2048),
        },
        in_features=['res3', 'res4', 'res5'],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm_layer=dict(type='GN', num_groups=32, num_channels=256)),
    transformer=dict(
        type='DINOTransformer',
        encoder=dict(
            type='DINOTransformerEncoder',
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels=num_feature_levels,
        ),
        decoder=dict(
            type='DINOTransformerDecoder',
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels=num_feature_levels,
        ),
        num_feature_levels=4,
        two_stage_num_proposals=num_queries,
    ),
    embed_dim=256,
    num_classes=num_classes,
    num_queries=num_queries,
    aux_loss=True,
    criterion=dict(
        type='DINOCriterion',
        num_classes=num_classes,
        matcher=dict(
            type='HungarianMatcher',
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type='focal_loss_cost',
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            'loss_class': 1,
            'loss_bbox': 5.0,
            'loss_giou': 2.0,
            'loss_class_dn': 1,
            'loss_bbox_dn': 5.0,
            'loss_giou_dn': 2.0,
        },
        loss_class_type='focal_loss',
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
    ),
    dn_number=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    device='cuda',
)

model = dict(
    type='Detectron2Wrapper',
    detector=detector,
    pretrained='detectron2://ImageNetPretrained/torchvision/R-50.pkl')

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(
        type='AdamW', lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
    # paramwise_cfg=dict(
    #     custom_cfg=dict(
    #         backbone=dict(lr_mult=0.1)
    #     )
    # )
)
