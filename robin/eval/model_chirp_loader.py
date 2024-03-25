import argparse
import os
import json
from tqdm import tqdm
import shortuuid

from robin.serve.robin_inference import Robin

def eval_model(args):
    # Model
    robin = Robin(args.model_path,
                 model_base=args.model_base,
                 device="cuda",
                 conv_mode=args.conv_mode,
                 temperature=args.temperature,
                 max_new_tokens=512
                )
    
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))

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
