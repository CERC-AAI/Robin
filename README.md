# Llava-Mistral
This is a fork of [github.com/haotian-liu/llava](https://github.com/haotian-liu/llava) to work with Mistral-type language models and OpenCLIP-SigLIP visual encoders. This repo and the accociated finetuned models were created in a collaboration between the AGI-Collective (specifically Kshitij Gupta, Daniel and Alexis) and Simon Ramstedt and with computing resources from [Hessian AI](https://hessian.ai/).

### Install
Ideally install into an empty venv (`python -m venv venv && source venv/bin/activate`)

```bash
pip install git+ssh://git@github.com/agi-collective/llava_mistral.git
```

### Run interactive command line interface
```bash
python -m llava.serve.cli \
    --model-path agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora \
    --model-base teknium/OpenHermes-2.5-Mistral-7B \
    --image-file https://llava-vl.github.io/static/images/view.jpg
```

### Use as library
```python
from llava.serve.pipeline import LlavaMistralPipeline

pipe = LlavaMistralPipeline(
    model_path="agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora",
    model_base="teknium/OpenHermes-2.5-Mistral-7B",
)

messages = [
  {"role": "USER", "content": "What's in the image?", "image": "https://llava-vl.github.io/static/images/view.jpg"},
]
messages = pipe(messages) 
# returns original messages list plus the new response, i.e.:
# {"role": "ASSISTANT", "content": ...}
```

### Available models

| Model                                                              | Base                              | GQA   | SQA Text | SQA Image |
| ------------------------------------------------------------------ | --------------------------------- | ----- | -------- | --------- |
| liuhaotian/llava-v1.5-7b                                           | lmsys/vicuna-7b-v1.5              | 62    | 70.43    | 66.8      |
| liuhaotian/llava-v1.5-13b                                          | lmsys/vicuna-7b-v1.5              | 63.3  |          | 71.6      |
| agi-collective/vicuna-7b-clip-finetune-lora                        | lmsys/vicuna-7b-v1.5              | **62.04** | 70.86    | 68.72     |
| agi-collective/vicuna-7b-siglip-so400m-finetune-lora               | lmsys/vicuna-7b-v1.5              | 56.79 | 68.76    | 67.48     |
| agi-collective/mistral-7b-siglip-so400m-finetune-lora              | mistralai/Mistral-7B-v0.1         | 49.44 | 73.66    | 68.57     |
| agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora           | teknium/OpenHermes-2.5-Mistral-7B | 54.48 | **79.56**    | **74.22**     |
| agi-collective/mistral-7b-oh-siglip-so400m-frozen-ve-finetune-lora | teknium/OpenHermes-2.5-Mistral-7B | 53.59 | 78.17    | 72.73     |

(best 7B model results **highlighted**)


### Evaluations
Evaluations were done using the a subset of the evaluation suite in https://github.com/haotian-liu/llava


### Training
For training details see the `pretrain.sh` and `finetune_lora.sh` scripts inside `scripts/*/`. The training code is for information purposes only and we won't be able help people debug problems with it for this preliminary release. More rigorous and less rushed and messy training scripts will be released in the future.

### Results
https://llava-vl.github.io/static/images/view.jpg
What's in this image?
Llava7b:
The image features a pier extending out over a large body of water, likely a lake. The pier is made of wood and has a bench at the end, providing a place for people to sit and enjoy the view. The water appears calm and serene, making it an ideal spot for relaxation.
HermesSiglip:
The image features a wooden pier extending out into a large body of water. The pier is surrounded by a serene and calm lake, providing a peaceful atmosphere. The pier appears to be empty, with no people or boats visible in the scene. The water is calm and still, creating a tranquil environment for those who visit the pier.

Desribe this image in detail:
Llava7b:
In the background, there are mountains visible, adding to the beauty of the landscape. The pier appears to be a popular spot for relaxation and enjoying the view, as there are several people scattered around the area, some closer to the water and others further away.
HermesSiglip:
The image features a wooden dock extending out into a large body of water. The dock is surrounded by a serene landscape, with a mountain visible in the background. The water appears calm and inviting, creating a peaceful atmosphere.
There are no people or boats visible in the scene, adding to the tranquility of the setting. The dock itself is wooden and appears to be well-maintained, providing a perfect spot for relaxation or fishing


Collie image:
What's in this image?
LlavaBase:
The image features a large, fluffy dog with a mix of brown and white fur sitting on a sidewalk. The dog is looking at the camera, possibly posing for a picture. The scene takes place in front of a brick wall, giving the impression of a residential area.
HermesSiglip:
The image features a large, brown and white dog sitting on a sidewalk. The dog is looking up at the camera, creating a friendly and approachable atmosphere. The dog is positioned near the center of the scene, with a brick wall in the background. The sidewalk is lined with grass, adding a natural touch to the setting.

What's in this image? Describe in detail.
Llava7b:
The image features a large, fluffy dog with a mix of brown and white fur, sitting on a sidewalk. The dog appears to be a Collie, and it is looking at the camera. The dog is positioned near a brick wall, which adds a contrasting element to the scene.
In the background, there is a car parked further away, and a person can be seen standing near the car. The overall atmosphere of the image is calm and serene, with the dog being the main focus of the scene.
HermesSiglip:
The image features a brown and white dog sitting on a sidewalk near a brick wall. The dog is looking at the camera, creating a friendly and approachable atmosphere. The dog appears to be a collie, which is a breed known for its long fur and fluffy appearance. The dog is sitting on the sidewalk, occupying a significant portion of the scene, and seems to be enjoying its time outdoors.


### Acknowledgements 
We would like to thank Hessian AI for providing us with free access to 8-16 A100 GPUs for a few weeks and to Florian and Patrick at Hessian AI for their free tech support.
