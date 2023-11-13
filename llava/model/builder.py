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

    if True:#'llava' in model_name.lower():#I think we still want to always use this branch
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
            
        if True:#'lora' in model_name.lower() and model_base is not None:#We are always loading from here for now... no option for not-lora for now.
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base)
            print('Loading LLaVA from base model...')
            
            #Temp
            if 'mistral' in model_name:
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, 
                    low_cpu_mem_usage=True, 
                    config=lora_cfg_pretrained, 
                    **kwargs,
                    # use_flash_attention_2 = True,
                )
            
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                print("Found lora_trainables")
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                print("Failed to find lora, exiting")
                exit()
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                
            #Right here we load the projection layer ONLY. WE DO NOT HAVE A VISION MODEL HERE.
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')

    image_processor = None

    if True:#'llava' in model_name.lower():#I'm pretty sure we always want to go with this branch
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
        
        #And this actually loads 'something'
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor        
    
    finetuned_ve = True
    if finetuned_ve:
        #We need to load the acutal weights into our vision_tower.
        original_weights = torch.load(model_path + "/non_lora_trainables.bin")
        #Convert names
        new_weights = {}
        for key in original_weights.keys():
            new_key = str(key).replace("base_model.model.model.vision_tower.","")
            new_weights[new_key] = original_weights[key]
        del original_weights
        
        #This is alread loaded.
        projection = {}
        #Doing it manually rn so that we can load strict no worries
        projections = ["base_model.model.model.mm_projector.0.weight", "base_model.model.model.mm_projector.0.bias", "base_model.model.model.mm_projector.2.weight", "base_model.model.model.mm_projector.2.bias"]
        for key in projections:
            projection[key] = new_weights.pop(key)
        
        
        result = vision_tower.load_state_dict(new_weights, strict = True)   
        print(result)
        
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    # print(model)
    return tokenizer, model, image_processor, context_len
