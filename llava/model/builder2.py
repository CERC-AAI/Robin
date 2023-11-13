import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):

    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

  
    lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    print('Loading LLaVA from base model...')
    
    LM = LlavaMistralForCausalLM if 'mistral' in model_name else LlavaLlamaForCausalLM

    model = LM.from_pretrained(
        model_base, 
        low_cpu_mem_usage=True, 
        config=lora_cfg_pretrained, 
        **kwargs,
        use_flash_attention_2 = True,
    )


    # TODO: what's this?
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))


    # Load LORA
    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')

    image_processor = None  # TODO: what's this?

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    #  This actually adds the vision tower to the model.
    vision_tower = model.get_vision_tower()
    print("Loaded this vision tower")
    
    #  And this actually loads 'something'
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor        
    

    #  We need to load the acutal weights into our vision_tower.
    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')

    #  Convert names
    vision_tower_weights = {}
    extra_model_weights = {}

    for key in list(non_lora_trainables.keys()):
        assert key.startswith("base_model.model.model.")

        new_key = key[len("base_model.model.model."):]

        if new_key.startswith("mm_projector."):
            extra_model_weights[new_key] = non_lora_trainables.pop(key)
        
        elif new_key.startswith("vision_tower."):
            # vision_tower_weights[new_key] = non_lora_trainables.pop(key)
            vision_tower_weights[new_key[len("vision_tower."):]] = non_lora_trainables.pop(key)
        else:
            assert False, f"wrong key {key}"
        

    assert len(non_lora_trainables) == 0, f"left over weights {non_lora_trainables.keys()}"

    model.load_state_dict(extra_model_weights, strict=False)
    vision_tower.load_state_dict(vision_tower_weights, strict=True)   
        
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    # print(model)
    return tokenizer, model, image_processor, context_len
