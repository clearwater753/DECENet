# model settings
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DECENet'),
    decode_head=dict(
        type='DECEHead',
        in_channels=128,
        channels=64,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='OhemCrossEntropy', loss_weight=1.0, class_weight=[
                0.4,
                0.7401073467900458,
                0.4,
                2.186435892962158,
                1.4180226829412388,
                1.5148512381543409,
                3.1669915504792465,
                2.528632458903019,
                0.4,
                1.4190945879881776,
                0.4,
                2.8273522618648497,
                5.32819868314015,
                0.4,
                1.4791393492191658,
                2.037008269639107,
                5.738221567757508,
                5.173006762881776,
                4.416876161714541,
            ],
            )),    
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 1024), stride=(360,320)))
