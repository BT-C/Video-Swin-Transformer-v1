_base_ = [
    './ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py'
]

# model settings
# load_from = '/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/weight/vmz_ipcsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-c3be9793.pth'
model = dict(
    backbone=dict(
        norm_eval=True,
        bn_frozen=True,
        bottleneck_mode='ip',
        pretrained=  # noqa: E251
        # 'https://download.openmmlab.com/mmaction/recognition/csn/ipcsn_from_scratch_r152_ig65m_20210617-c4b99d38.pth'  # noqa: E501
        '/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/weight/vmz_ipcsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-c3be9793.pth'
    ))

work_dir = './work_dirs/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb'  # noqa: E501
