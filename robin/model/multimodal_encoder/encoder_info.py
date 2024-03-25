# file used to detail which encoder class to use with each model
#vit_large_patch14_clip_224.openai 
#vit_large_patch14_reg4_dinov2.lvd142m
CLIP_COMPATIBLE_MODELS = [
    'metaclip-h14-fullcc2.5b', 
    'metaclip-l14-fullcc2.5b'

    'clip-vit-bigg-14-laion2b-39b-b160k', 
    'CLIP-ViT-L-14-laion2B-s32B-b82K',
    'metaclip-l14-fullcc2.5b', 

    'clip-vit-large-patch14',
    'clip-vit-large-patch14-336',
    'vit_large_patch14_clip_224.openai',
]

OPENCLIP_CONFIG_MAP_tmp = {
    'eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k': 'EVA02-E-14-plus',
    'eva02_large_patch14_clip_224.merged2b_s4b_b131k' : 'EVA02-L-14',
    'ViT-SO400M-14-SigLIP-384': 'ViT-SO400M-14-SigLIP-384',
    'ViT-L-16-SigLIP-256': 'ViT-L-16-SigLIP-256',
    'ViT-bigG-14-CLIPA-336-datacomp1B': 'ViT-bigG-14-CLIPA-336',
    'ViT-bigG-14-CLIPA-datacomp1B': 'ViT-bigG-14-CLIPA',
    'ViT-L-14-CLIPA-datacomp1B' : 'ViT-L-14-CLIPA',
    'ViT-L-14-CLIPA-336-datacomp1B' : 'ViT-L-14-CLIPA-336',
    'DFN5B-CLIP-ViT-H-14': 'ViT-H-14',
    'DFN2B-CLIP-ViT-L-14': 'ViT-L-14',
    'CLIP-ViT-B-16-laion2B-s34B-b88K': 'ViT-B-16',
    'CLIP-ViT-L-14-laion2B-s32B-b82K': 'ViT-L-14',
    'CLIP-ViT-H-14-laion2B-s32B-b79K': 'ViT-H-14',
    'CLIP-ViT-g-14-laion2B-s34B-b88K': 'ViT-g-14',
}

TIMM_ON_OPENCLIP = [
    'eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k',
    'ViT-SO400M-14-SigLIP-384',
    'ViT-L-16-SigLIP-256',
    'eva02_large_patch14_clip_224.merged2b_s4b_b131k',
]


CLIP_COMPATIBLE_MODELS = [m.lower() for m in CLIP_COMPATIBLE_MODELS]
TIMM_ON_OPENCLIP = [m.lower() for m in TIMM_ON_OPENCLIP]
OPENCLIP_CONFIG_MAP = OPENCLIP_CONFIG_MAP_tmp.copy()
for k, v in OPENCLIP_CONFIG_MAP_tmp.items():
    OPENCLIP_CONFIG_MAP[k.lower()] = v


# OpenCLIP models must be associated with a config from this list:
# coca_base
# coca_roberta-ViT-B-32
# coca_ViT-B-32
# coca_ViT-L-14
# convnext_base
# convnext_base_w
# convnext_base_w_320
# convnext_large
# convnext_large_d
# convnext_large_d_320
# convnext_small
# convnext_tiny
# convnext_xlarge
# convnext_xxlarge
# convnext_xxlarge_320
# EVA01-g-14
# EVA01-g-14-plus
# EVA02-B-16
# EVA02-E-14
# EVA02-E-14-plus
# EVA02-L-14
# EVA02-L-14-336
# mt5-base-ViT-B-32
# mt5-xl-ViT-H-14
# nllb-clip-base
# nllb-clip-large
# RN50
# RN50-quickgelu
# RN50x4
# RN50x16
# RN50x64
# RN101
# RN101-quickgelu
# roberta-ViT-B-32
# swin_base_patch4_window7_224
# ViT-B-16
# ViT-B-16-plus
# ViT-B-16-plus-240
# ViT-B-16-quickgelu
# ViT-B-16-SigLIP
# ViT-B-16-SigLIP-256
# ViT-B-16-SigLIP-384
# ViT-B-16-SigLIP-512
# ViT-B-16-SigLIP-i18n-256
# ViT-B-32
# ViT-B-32-256
# ViT-B-32-plus-256
# ViT-B-32-quickgelu
# ViT-bigG-14
# ViT-bigG-14-CLIPA
# ViT-bigG-14-CLIPA-336
# ViT-e-14
# ViT-g-14
# ViT-H-14
# ViT-H-14-378-quickgelu
# ViT-H-14-CLIPA
# ViT-H-14-CLIPA-336
# ViT-H-14-quickgelu
# ViT-H-16
# ViT-L-14
# ViT-L-14-280
# ViT-L-14-336
# ViT-L-14-CLIPA
# ViT-L-14-CLIPA-336
# ViT-L-14-quickgelu
# ViT-L-16
# ViT-L-16-320
# ViT-L-16-SigLIP-256
# ViT-L-16-SigLIP-384
# ViT-M-16
# ViT-M-16-alt
# ViT-M-32
# ViT-M-32-alt
# ViT-S-16
# ViT-S-16-alt
# ViT-S-32
# ViT-S-32-alt
# ViT-SO400M-14-SigLIP
# ViT-SO400M-14-SigLIP-384
# vit_medium_patch16_gap_256
# vit_relpos_medium_patch16_cls_224
# xlm-roberta-base-ViT-B-32
# xlm-roberta-large-ViT-H-14