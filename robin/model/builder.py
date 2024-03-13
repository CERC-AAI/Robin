import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from robin.model import *
from robin.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


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

    # Load LLaVA model
    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        
    if model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base)
        print('Loading LLaVA from base model...')
        
        if 'mistral' in model_name.lower():
            model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        elif any(x in model_name.lower() for x in ['neox', 'pythia', 'hi-nolin']):
            model = LlavaGPTNeoXForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base, 
                low_cpu_mem_usage=True, 
                config=lora_cfg_pretrained, 
                **kwargs,
                # use_flash_attention_2 = True,
            )
        if any(x in model_name.lower() for x in ['neox', 'pythia', 'hi-nolin']):  # neox has different names for modules
            token_num, tokem_dim = model.embed_out.out_features, model.embed_out.in_features
            if model.embed_out.weight.shape[0] != token_num:
                model.embed_out.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        else:
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional LLaVA weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            print("Found non_lora_trainables")
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')
            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables) or any(k.startswith('model.gpt_neox.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path, 
            low_cpu_mem_usage = True, 
            **kwargs,
            use_flash_attention_2 = False,
        )

    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    #This actually adds the vision tower to the model.
    vision_tower = model.get_vision_tower()
    print("Loaded this vision tower")
    
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor        
    
    finetuned_ve = False if ("frozen" in model_name.lower() or "llava" in model_name.lower()) else True
    if finetuned_ve:
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            print("Found lora_trainables")
            original_weights = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'))
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file)
            original_weights = load_from_hf(model_path, 'non_lora_trainables.bin')

        #Convert names
        new_weights = {}
        for key in original_weights.keys():
            new_key = str(key).replace("base_model.model.model.vision_tower.","")
            new_key = str(key).replace("base_model.model.gpt_neox.vision_tower.","")
            new_weights[new_key] = original_weights[key]
        del original_weights
        
        
        #This is so that we can load strict
        projection = {}
        projections = ["base_model.model.model.mm_projector.0.weight", "base_model.model.model.mm_projector.0.bias", "base_model.model.model.mm_projector.2.weight", "base_model.model.model.mm_projector.2.bias"]
        projections = [item.replace('model.mm_projector', 'gpt_neox.mm_projector') for item in projections]
        for key in projections:
            if key in new_weights:
                projection[key] = new_weights.pop(key)
        
        result = vision_tower.load_state_dict(new_weights, strict = True)   
        print("Loading strict resuts:", result)
        
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
