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