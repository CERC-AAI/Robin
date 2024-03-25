import argparse
import torch
import os
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

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Model
    robin = Robin(args.model_path,
                 model_base=args.model_base,
                 device="cuda",
                 conv_mode=args.conv_mode,
                 temperature=args.temperature,
                 max_new_tokens=128
                )

#    print("questions", args.question_file)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    #New
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    img_folder = os.path.expanduser(args.image_folder)

    for line in tqdm(questions):
        idx = line["question_id"]
        cur_prompt = line["text"]
        image_file = os.path.join(img_folder, line["image"])

        outputs = robin(image_file, cur_prompt)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": robin.model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


def eval_model_old(args):
    # Model
    robin = Robin(args.model_path,
                 model_base=args.model_base,
                 device="cuda",
                 conv_mode=args.conv_mode,
                 temperature=args.temperature,
                 max_new_tokens=128
                )
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        
        outputs = robin(image_file, qs)


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": outputs,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": robin.model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
