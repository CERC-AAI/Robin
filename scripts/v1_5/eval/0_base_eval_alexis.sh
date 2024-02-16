

#/pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/trained_models/vicuna-7b-siglip-so400m-finetune-lora

#We need to pass over the model path and base model path dir thing..


# --model-path /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/checkpoints/llava-v1.5-7b-lora3 \
# --model-base /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b

export TRANSFORMERS_CACHE=/localdisks/rogeralexis/downloaded_models/hf_cache
export HF_HOME=/localdisks/rogeralexis/downloaded_models/hf_cache

model_path="/home/rogeralexis/scratch/checkpoints/robin_v2_1/finetune_VEtrue"
model_path="agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora"
#apple/DFN5B-CLIP-ViT-H-14
model_base="teknium/OpenHermes-2.5-Mistral-7B"

version="unknownstuff"
./sqa_daniel.sh $model_path $model_base $version 


#Base
# ./sqa.sh /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/checkpoints/llava-v1.5-7b-lora3 /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b base_ve
# ./textvqa.sh /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/checkpoints/llava-v1.5-7b-lora3 /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b base_ve
# ./gqa.sh /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/checkpoints/llava-v1.5-7b-lora3 /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b base_ve
# ./vqav2.sh /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/checkpoints/llava-v1.5-7b-lora3 /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b base_ve
# ./vizwiz.sh /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/checkpoints/llava-v1.5-7b-lora3 /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b base_ve
# ./mmbench.sh /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/checkpoints/llava-v1.5-7b-lora3 /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b base_ve
# ./mmbench_cn.sh /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/checkpoints/llava-v1.5-7b-lora3 /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b base_ve
# ./mmvet.sh /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/checkpoints/llava-v1.5-7b-lora3 /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b base_ve
# ./mme.sh /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/checkpoints/llava-v1.5-7b-lora3 /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b base_ve
# ./llavabench.sh /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/checkpoints/llava-v1.5-7b-lora3 /pfss/mlde/workspaces/mlde_wsp_Ramstedt_Mila/hf/vicuna-7b base_ve
