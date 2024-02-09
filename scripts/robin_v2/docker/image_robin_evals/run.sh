#	-it \
docker run \
	--rm \
	--detach \
	--gpus 1 \
	--shm-size 8G \
	--volume /home/$(whoami)/checkpoints:/export \
	docker.io/library/robin_evals \
		agi-collective/mistral-7b-oh-siglip-so400m-finetune-lora \
		teknium/OpenHermes-2.5-Mistral-7B \
