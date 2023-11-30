# Robin-Llava

**[Llava](https://github.com/haotian-liu/llava)-fork enabling Mistral & Open-Hermes-2.5 language models to process images**

This repo and the accociated finetuned models were created in a collaboration between the AGI-Collective and Simon Ramstedt, with computing resources from [Hessian AI](https://hessian.ai/).

### Install
Ideally install into an empty venv (`python -m venv venv && source venv/bin/activate`)

```bash
pip install git+ssh://git@github.com/agi-collective/robin_llava.git
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
We use the following components: 
- Base LLM: We explore using Mistral and OpenHermes-2.5
- Base Vision Model: We use the SigLIP model which gives stronger performance on vision benchmarks compared to CLIP 
- We finetune the Vision Encoder hoping the next token prediction loss helps further improves the vision capabilities of the pretrained vision encoder

| Model                                                              | Base                              | GQA   | SQA Text | SQA Image |
| ------------------------------------------------------------------ | --------------------------------- | ----- | -------- | --------- |
| liuhaotian/llava-v1.5-7b                                           | lmsys/vicuna-7b-v1.5              | 62    | 70.43    | 66.8      |
| liuhaotian/llava-v1.5-13b                                          | lmsys/vicuna-7b-v1.5              | 63.3  |          | 71.6      |
| agi-collective/vicuna-7b-clip-finetune-lora                        | lmsys/vicuna-7b-v1.5              | **62.04** | 70.86    | 68.72     |
| agi-collective/vicuna-7b-siglip-so400m-finetune-lora               | lmsys/vicuna-7b-v1.5              | 56.79 | 68.76    | 67.48     |
| agi-collective/mistral-7b-siglip-so400m-finetune-lora              | mistralai/Mistral-7B-v0.1         | 49.44 | 73.66    | 68.57     |
| agi-collective/mistral-7b-oh-siglip-so400m-frozen-ve-finetune-lora | teknium/OpenHermes-2.5-Mistral-7B | 53.59 | 78.17    | 72.73     |
| agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora           | teknium/OpenHermes-2.5-Mistral-7B | 54.48 | **79.56**    | **74.22**     |


(best 7B model results **highlighted**)


### Evaluations
Evaluations were done using the a subset of the evaluation suite in https://github.com/haotian-liu/llava


### Training
For training details see the `pretrain.sh` and `finetune_lora.sh` scripts inside `scripts/*/`. The training code is for information purposes only and we won't be able help people debug problems with it for this preliminary release. More rigorous and less rushed and messy training scripts will be released in the future.

### Example 1
![](https://llava-vl.github.io/static/images/view.jpg)
**User:** What's in this image?

**Llava7b:**
The image features a pier extending out over a large body of water, likely a lake. The pier is made of wood and has a bench at the end, providing a place for people to sit and enjoy the view. The water appears calm and serene, making it an ideal spot for relaxation.

**HermesSiglip (Ours):**
The image features a wooden pier extending out into a large body of water. The pier is surrounded by a serene and calm lake, providing a peaceful atmosphere. The pier appears to be empty, with no people or boats visible in the scene. The water is calm and still, creating a tranquil environment for those who visit the pier.

**User:** Desribe this image in detail:

**Llava7b:**
In the background, there are mountains visible, adding to the beauty of the landscape. The pier appears to be a popular spot for relaxation and enjoying the view, as there are several people scattered around the area, some closer to the water and others further away.

**HermesSiglip (Ours):**
The image features a wooden dock extending out into a large body of water. The dock is surrounded by a serene landscape, with a mountain visible in the background. The water appears calm and inviting, creating a peaceful atmosphere.
There are no people or boats visible in the scene, adding to the tranquility of the setting. The dock itself is wooden and appears to be well-maintained, providing a perfect spot for relaxation or fishing

### Example 2
![](https://pbs.twimg.com/media/GAL3KpGaAAAwNUg?format=jpg&name=360x360)

**User:** What is written in this image?

**Llava7b:**
The words "No one grows ketchup like Heinz" are written in this image.

 **HermesSiglip (Ours):**
In this image, the words "No one grows ketchup like Heinz" are written on the side of a bottle, which is a Heinz ketchup bottle.

### Example 3
![]([https://pbs.twimg.com/media/GAL3KpGaAAAwNUg?format=jpg&name=360x360](https://scienceqa-webdemo.s3.us-west-1.amazonaws.com/extra/5727/image.png))

**User:** Will these magnets attract or repel each other?

**Llava7b:**
The magnets in the image are of different colors and placed next to each other. It is not possible to determine their polarity from the image alone, so it is not possible to definitively say whether they will attract or repel each other. The behavior of magnets depends on their polarity and the orientation of their magnetic fields. To determine their interaction, one would need to observe the magnets in a more controlled environment, such as a magnetic field tester or a magnetic compass.

 **HermesSiglip (Ours):**
In the image, there are two magnets placed on a gray background, one with a green stripe and the other with a blue stripe. Since the magnets have opposite colors, they will attract each other due to their magnetic properties. The magnets are aligned in a way that their magnetic poles are facing each other, which causes them to be attracted to one another. This phenomenon is known as the magnetic attraction between two oppositely charged magnets.

### Acknowledgements 
We would like to thank Hessian AI for providing us with free access to 8-16 A100 GPUs for a few weeks and to Florian and Patrick at Hessian AI for their support.

 <a style="" src=""><img style="" src="https://hessian.ai/wp-content/themes/hessianai/img/hessian-ai-logo.svg"></img></a>
