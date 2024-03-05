import os
import argparse
import json
import re
import pandas as pd

from tqdm import tqdm

NUMBERS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

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
        ground_truth_val = question_df[question_df["question_index"] == question_idx]["answer"].values[0].lower()

        # Convert to text if ground truth is an integer
        try:
            num = int(ground_truth_val)
            ground_truth_val = NUMBERS[num]
        except Exception as e:
            pass

        yesno_match =  re.search(r'^(yes|no)', pred_val)
        next_word_matches = re.finditer(r'(?:is made of|is a|has a|there are|made of|is) (\w+)', pred_val)
        
        if yesno_match:
            pred_word = yesno_match.group(1)
            if ground_truth_val == pred_word:
                num_correct += 1
                correct_list.append([ground_truth_val, pred_val])
            else:
                wrong_list.append([ground_truth_val, pred_val])
        
        # Find all words that come right after 'is a', 'there are', 'is', 'made of'
        elif next_word_matches:
            found = False
            for next_word_match in next_word_matches:
                pred_word = next_word_match.group(1)

                if ground_truth_val == pred_word:
                    num_correct += 1
                    found = True
                    correct_list.append([ground_truth_val, pred_val])
                    break

            if not found:
                wrong_list.append([ground_truth_val, pred_val])    

        else:
            wrong_list.append([ground_truth_val, pred_val])

    acc = round(num_correct / num_total, 5)
    print(f"Accuracy: {acc} ; {num_correct} / {num_total} correct.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="CLEVR_v1.0/questions/CLEVR_test_questions.json")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    args = parser.parse_args()

    generate_score(args)