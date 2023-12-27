# file used to detail which encoder class to use with each model

CLIP_COMPATIBLE_MODELS = [
    'metaclip-h14-fullcc2.5b', 
    'clip-vit-bigg-14-laion2b-39b-b160k', 
    'metaclip-l14-fullcc2.5b', 
    'clip-vit-large-patch14-336'
]


OPENCLIP_CONFIG_MAP = {
    'eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k': 'EVA02-E-14-plus',
    'ViT-SO400M-14-SigLIP-384': 'ViT-SO400M-14-SigLIP-384',
    'ViT-bigG-14-CLIPA-336-datacomp1B': 'ViT-bigG-14-CLIPA-336',
    'ViT-bigG-14-CLIPA-datacomp1B': 'ViT-bigG-14-CLIPA',
    'DFN5B-CLIP-ViT-H-14': 'ViT-H-14' 
}