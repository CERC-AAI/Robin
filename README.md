# Robin-LLaVA

**A LLaVA-fork enabling the Mistral-7B & Open-Hermes-2.5 language models to process images**

This repo and the associated models were created in collaboration between the Robin team at AGI-Collective and Simon Ramstedt, with computing resources from [Hessian-AI](https://hessian.ai/) and [OLCF](https://www.olcf.ornl.gov/).

The Robin Effort: The goal of this effort is to study the different components we can use when merging pretrained vision and language models, and to build SoTA visual understanding models.

As part of the first milestone we study the pretrained LLMs (Vicuna, Mistral and OpenHermes 2.5) and Vision models (CLIP and SigLIP), further improving capabilities by finetuning the vision encoder. 


## Installation
Ideally install into an empty venv (`python -m venv venv && source venv/bin/activate`)

```bash
pip install git+ssh://git@github.com/agi-collective/robin_llava.git
```

Note that currently Robin-LlaVA inference requires a Nvidia GPU with 24GB+ since CPU inference and quantized inference don't work reliably yet. Training was done on a machine with 8 x A100 80GB Nvidia GPUs provided by [Hessian-AI](https://hessian.ai/).

## Run interactive command line interface
```bash
python -m llava.serve.cli \
    --model-path agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora \
    --model-base teknium/OpenHermes-2.5-Mistral-7B \
    --image-file https://llava-vl.github.io/static/images/view.jpg
```

## Use as library
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

## Available models
We use the following components: 
- **Base LLM**: We explore using Vicuna, Mistral and OpenHermes-2.5
- **Base Vision Model**: We use the SigLIP model since it gives stronger performance on vision benchmarks compared to CLIP 
- We finetune the Vision Encoder hoping the next token prediction loss helps further improves the vision capabilities of the pretrained vision encoder

| Model                                                              | Base                              | GQA   | SQA Text | SQA Image |
| ------------------------------------------------------------------ | --------------------------------- | ----- | -------- | --------- |
| liuhaotian/llava-v1.5-7b                                           | lmsys/vicuna-7b-v1.5              | 62    | 70.43    | 66.8      |
| liuhaotian/llava-v1.5-13b                                          | lmsys/vicuna-7b-v1.5              | 63.3  |          | 71.6      |
| [agi-collective/vicuna-7b-clip-finetune-lora](https://huggingface.co/agi-collective/vicuna-7b-clip-finetune-lora)                         | lmsys/vicuna-7b-v1.5              | **62.04** | 70.86    | 68.72     |
| [agi-collective/vicuna-7b-siglip-so400m-finetune-lora](https://huggingface.co/agi-collective/vicuna-7b-siglip-so400m-finetune-lora)               | lmsys/vicuna-7b-v1.5              | 56.79 | 68.76    | 67.48     |
| [agi-collective/mistral-7b-siglip-so400m-finetune-lora](https://huggingface.co/agi-collective/mistral-7b-siglip-so400m-finetune-lora)              | mistralai/Mistral-7B-v0.1         | 49.44 | 73.66    | 68.57     |
| [agi-collective/mistral-7b-oh-siglip-so400m-frozen-ve-finetune-lora](https://huggingface.co/agi-collective/mistral-7b-oh-siglip-so400m-frozen-ve-finetune-lora) | teknium/OpenHermes-2.5-Mistral-7B | 53.59 | 78.17    | 72.73     |
| [agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora](https://huggingface.co/agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora)           | teknium/OpenHermes-2.5-Mistral-7B | 54.48 | **79.56**    | **74.22**     |

(best 7B model results **highlighted**)


## Evaluations
Evaluations were done using the a subset of the evaluation suite in https://github.com/haotian-liu/llava. More evaluations including both vision language and language only benchmarks coming soon!


## Training
For training details see the `pretrain.sh` and `finetune_lora.sh` scripts inside `scripts/*/`. The training code is for information purposes only and we won't be able help people debug problems with it for this preliminary release. More rigorous and less rushed and messy training scripts will be released in the future.

# Examples
## Example 1
![](https://llava-vl.github.io/static/images/view.jpg)
**User:** What's in this image?

**Llava7b:**
The image features a pier extending out over a large body of water, likely a lake. The pier is made of wood and has a bench at the end, providing a place for people to sit and enjoy the view. The water appears calm and serene, making it an ideal spot for relaxation.

**HermesSiglip (Ours):**
The image features a wooden pier extending out into a large body of water. The pier is surrounded by a serene and calm lake, providing a peaceful atmosphere. The pier appears to be empty, with no people or boats visible in the scene. The water is calm and still, creating a tranquil environment for those who visit the pier.

**User:** Describe this image in detail:

**Llava7b:**
In the background, there are mountains visible, adding to the beauty of the landscape. The pier appears to be a popular spot for relaxation and enjoying the view, as there are several people scattered around the area, some closer to the water and others further away.

**HermesSiglip (Ours):**
The image features a wooden dock extending out into a large body of water. The dock is surrounded by a serene landscape, with a mountain visible in the background. The water appears calm and inviting, creating a peaceful atmosphere.
There are no people or boats visible in the scene, adding to the tranquility of the setting. The dock itself is wooden and appears to be well-maintained, providing a perfect spot for relaxation or fishing

## Example 2
![](https://scienceqa-webdemo.s3.us-west-1.amazonaws.com/extra/5727/image.png)

**User:** Will these magnets attract or repel each other?

**Llava7b:**
The magnets in the image are of different colors and placed next to each other. It is not possible to determine their polarity from the image alone, so it is not possible to definitively say whether they will attract or repel each other. The behavior of magnets depends on their polarity and the orientation of their magnetic fields. To determine their interaction, one would need to observe the magnets in a more controlled environment, such as a magnetic field tester or a magnetic compass.

 **HermesSiglip (Ours):**
In the image, there are two magnets placed on a gray background, one with a green stripe and the other with a blue stripe. Since the magnets have opposite colors, they will attract each other due to their magnetic properties. The magnets are aligned in a way that their magnetic poles are facing each other, which causes them to be attracted to one another. This phenomenon is known as the magnetic attraction between two oppositely charged magnets.

## Example 3
![](https://github.com/AGI-Collective/Robin-LLaVA/blob/robin-main/images/ILSVRC2012_val_00000003.JPEG)

**User:** `What's in this image? Describe in detail.`

**Llava7b:**
The image features a large, fluffy dog with a mix of brown and white fur, sitting on a sidewalk. The dog appears to be a Collie, and it is looking at the camera. The dog is positioned near a brick wall, which adds a contrasting element to the scene.
In the background, there is a car parked further away, and a person can be seen standing near the car. The overall atmosphere of the image is calm and serene, with the dog being the main focus of the scene.

**HermesSiglip (Ours):**
The image features a brown and white dog sitting on a sidewalk near a brick wall. The dog is looking at the camera, creating a friendly and approachable atmosphere. The dog appears to be a collie, which is a breed known for its long fur and fluffy appearance. The dog is sitting on the sidewalk, occupying a significant portion of the scene, and seems to be enjoying its time outdoors.

## Limitations
For the Open-Hermes-2.5-based model we switched away from the ChatML chat template and back to the Llava default Vicuna chat template. The benchmarks and example results below show that this worked fine but we do believe that we can achieve even better performance and usability by retraining the model with ChatML.

### Contributors:
**Robin Team @ AGI-Collective**: Alexis Roger, Andrew R Williams, Daniel Kaplan, Edwin Fennell, George Adamopoulos, Kshitij Gupta, Prateek Humane, Quentin Anthony, Rishika Bhagwatkar, Sun Qi, Yuchen Lu, Irina Rish

**Simon Ramstedt**: https://simonramstedt.com/

### Acknowledgements 
We would like to thank Hessian-AI for providing us with free access to 8-16 A100 GPUs for a few weeks and to Florian and Patrick at Hessian AI for their support. We would also like to thank Oak Ridge Leadership Computing Facility (OLCF), the DOE Office of Science User Facility. Prelimnary experiments were conducted on the INCITE compute grant on Summit supercomputer supported under Contract DE-AC05-00OR22725. This grant was awarded to AAI CERC lab for their Scalable Foundation Models for Transferrable Generalist AI project. This work was in collaboration with representatives from EleutherAI. The code in this repo is based on [github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA).

 <a style="" src=""><img style="" src="https://hessian.ai/wp-content/themes/hessianai/img/hessian-ai-logo.svg"></img></a>
 
 <a style="" src=""><img style="width: 40%; height: auto;" src="https://www.olcf.ornl.gov/wp-content/uploads/2016/07/olcf_retina_logo.png"></img></a>
