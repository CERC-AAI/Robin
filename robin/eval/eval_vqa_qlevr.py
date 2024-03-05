import os
import argparse
import json
import re
import pandas as pd

from tqdm import tqdm

def generate_score(args):
    question_filepath = args.question_file
    answer_filepath = args.answers_file
    
    with open(question_filepath, 'r') as f:
        question_data = json.load(f)

    question_df = pd.DataFrame(question_data["questions"])

    predicted_answers = [json.loads(line) for line in open(answer_filepath, 'r')]

    num_correct = 0
    num_total = len(predicted_answers)
    correct_list = []
    wrong_list = []

    for pred_ans in tqdm(predicted_answers):
        question_idx = pred_ans["question_id"]
        pred_val = pred_ans["text"].lower()
        ground_truth_val = question_df[question_df["question_index"] == question_idx]["answer"].values[0]

        yesno_match =  re.search(r'^(yes|no)', pred_val)
        
        if yesno_match:
            pred_word = yesno_match.group(1)
            pred_word = True if pred_word == 'yes' else False
            
            if ground_truth_val == pred_word:
                num_correct += 1
                correct_list.append([ground_truth_val, pred_val])
            else:
                wrong_list.append([ground_truth_val, pred_val])
        else:
            wrong_list.append([ground_truth_val, pred_val])

    acc = round(num_correct / num_total, 5)
    print(f"Accuracy: {acc} ; {num_correct} / {num_total} correct.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="questions_test.json")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    args = parser.parse_args()

    generate_score(args)