import argparse
from robin.serve.robin_inference import Robin


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora")
    parser.add_argument("--model-base", type=str, default="teknium/OpenHermes-2.5-Mistral-7B")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()


    robin = Robin(args.model_path, 
          model_base = args.model_base,
          device = args.device, 
          conv_mode = args.conv_mode, 
          temperature = args.temperature, 
          max_new_tokens = args.max_new_tokens, 
          load_8bit = args.load_8bit, 
          load_4bit = args.load_4bit, 
          debug = args.debug, 
          image_aspect_ratio = args.image_aspect_ratio,
          lazy_load = False)

    img_url = input("Enter the image URL: ")
    prompt = input("Enter the prompt: ")
    robin(img_url, prompt, streamer=True)