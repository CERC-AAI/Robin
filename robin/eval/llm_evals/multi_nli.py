import jsonlines
import numpy as np
from sklearn.metrics import accuracy_score
import argparse

from robin.serve.robin_inference import Robin

def classify_string(s):
    # Define the keywords to search for
    keywords = ['entailment', 'contradiction', 'neutral']
    
    # Check if the input string contains any of the keywords
    for keyword in keywords:
        if keyword in s.lower():  # Convert the string to lowercase to make the search case-insensitive
            return keyword
    
    # Return an empty string if none of the keywords are found
    return ''


# Assuming 'robin' is already instantiated as per your provided code

def evaluate_model(file_path, robin):
    # load in robi

    true_labels = []
    predicted_labels = []

    num = 0

    with jsonlines.open(file_path) as reader:
        for obj in reader:
            gold_label = obj['gold_label']
            sentence1 = obj['sentence1']
            sentence2 = obj['sentence2']
            
            # Skip examples without a valid gold label
            if gold_label not in ['entailment', 'contradiction', 'neutral']:
                continue

            # Use Robin for prediction
            # Here we assume a format for the input question. You might need to adjust this.
            question = f"What is the relationship between: '{sentence1}' and '{sentence2}'? ('entailment', 'contradiction', 'neutral')"
            outputs = robin(None, question)

            outputs = classify_string(outputs)

            # print(question)
            # print(outputs)
            
            # Assuming 'outputs' contains the predicted label. You might need to extract the label from 'outputs'.
            predicted_label = outputs

            true_labels.append(gold_label)
            predicted_labels.append(predicted_label)

            num += 1
            if num > 11:
                break

    # Calculate the accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/localdisks/rogeralexis/downloaded_models/mistral-7b-oh-siglip-so400m-finetune-lora")
    parser.add_argument("--model-base", type=str, default="/localdisks/rogeralexis/george_files/Robin/robin/eval/llm_evals/test_dir/OpenHermes-2.5-Mistral-7B")
    parser.add_argument("--eval-file", type=str, default="/localdisks/rogeralexis/george_files/multinli_1.0/multinli_1.0_dev_matched.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    robin = Robin(args.model_path,
                model_base=args.model_base,
                device="cuda",
                conv_mode="llava_v1",
                temperature=0.2,
                max_new_tokens=128
            )

    # file_path = '/localdisks/rogeralexis/george_files/multinli_1.0/multinli_1.0_dev_matched.jsonl'
    evaluate_model(args.eval_file, robin)
