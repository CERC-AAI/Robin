import os
import argparse
import torch
import json
from tqdm import tqdm
import shortuuid

from robin.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from robin.conversation import conv_templates, SeparatorStyle
from robin.model.builder import load_pretrained_model
from robin.utils import disable_torch_init
from robin.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

from robin.serve.robin_inference import Robin

# Note: We assuming that the question file passed in is the in the format provided in the CLEVR dataset downloaded from https://cs.stanford.edu/people/jcjohns/clevr/ 
# https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
# e.g. CLEVR_v1.0/questions/CLEVR_test_questions.json , which correspond to images in CLEVR_v1.0/images/test
def eval_model(args):
    questions_file_path = os.path.expanduser(args.question_file)
    img_folder = os.path.expanduser(args.image_folder)
    answers_file = os.path.expanduser(args.answers_file)
    question_data = [ json.loads(line) for line in open(os.path.expanduser(questions_file_path), "r")]
    questions = question_data[0]["questions"]

    ans_file = open(answers_file, "w")

    robin = Robin(args.model_path,
                model_base=args.model_base,
                device="cuda",
                conv_mode="llava_v1",
                temperature=0.2,
                max_new_tokens=128
            )

    for q in tqdm(questions):

        question_idx = q["question_index"]
        image_filename = q["image_filename"]
        image_path = os.path.join(img_folder, image_filename)
        question_prompt = q["question"]
        answer_id = shortuuid.uuid()

        # Inference
        outputs = robin(image_path, question_prompt)

        ans_file.write(json.dumps({"question_id": question_idx,
                                    "prompt": question_prompt,
                                    "text": outputs,
                                    "answer_id": answer_id,
                                    "model_id": robin.model_name,
                                    "metadata": {}}) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora")
    parser.add_argument("--model-base", type=str, default="teknium/OpenHermes-2.5-Mistral-7B")
    parser.add_argument("--image-folder", type=str, default="CLEVR_v1.0/images/test")
    parser.add_argument("--question-file", type=str, default="CLEVR_v1.0/questions/CLEVR_test_questions.json")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
