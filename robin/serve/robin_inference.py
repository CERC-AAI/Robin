import argparse
import torch
import os
import json

from robin.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from robin.conversation import conv_templates, SeparatorStyle
from robin.model.builder import load_pretrained_model
from robin.utils import disable_torch_init
from robin.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


class Robin:
    def __init__(self, 
                 model_path,
                 model_base="",
                 device="cuda",
                 conv_mode="vicuna_v1",
                 temperature=0.2,
                 max_new_tokens=512,
                 load_8bit=False,
                 load_4bit=False,
                 debug=False,
                 image_aspect_ratio='pad',
                 lazy_load=False):
        
        self.model_path = os.path.expanduser(model_path)
        self.model_base = model_base
        self.device = device
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.debug = debug
        self.image_aspect_ratio = image_aspect_ratio

        self.loaded = False
        if not lazy_load:
            self.load_model()

    def find_base(self):
        # TODO
        raise "Base model arg in config not implemented yet!"
        if os.path.exists(self.model_path):
            with open(os.path.join(self.model_path, "config.json"), "r") as f:
                config = json.load(f)
                print("CONFIG ", config)
                self.model_base = config["_name_or_path"]
        else:
            print("No local model found, trying to download from Hugging Face")
            try:
                print("Trying to download model from Hugging Face", self.model_path, '...')
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(self.model_path)
                self.model_base = config._name_or_path
                print("CONFIG ", config)
                print("Model found at", self.model_base)
            except Exception as e:
                print("Failed to download model from Hugging Face", self.model_path, e)
                self.model_base = self.model_path

    def load_model(self):
        disable_torch_init()

        self.model_name = get_model_name_from_path(self.model_path)

        if len(self.model_base) == 0:
            self.find_base()
        
        print(f"Loading model {self.model_name} from {self.model_base}...")

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, self.model_base, self.model_name, self.load_8bit, self.load_4bit, device=self.device)
        
        self.loaded = True

    def load_image(self, image_file):
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    def __call__(self, img_url, prompt):
        if not self.loaded:
            self.load_model()

        conv = conv_templates[self.conv_mode].copy()
        roles = conv.roles

        if img_url is not None:
            image = self.load_image(img_url)

            # Similar operation in model_worker.py
            image_tensor = process_images([image], self.image_processor, self.image_aspect_ratio)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        else:
            image_tensor = None

        if self.debug: print(f"{roles[1]}: ", end="")

        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)] if conv.version == "v0" else None
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=stopping_criteria)

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        conv.messages[-1][-1] = outputs

        if self.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        return outputs

