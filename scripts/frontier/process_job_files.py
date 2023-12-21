import json
import matplotlib.pyplot as plt

job_name = "robin-1543998"
parameters = [
    'Pretrain',
    'LLM: OpenHermes-2.5-Mistral-7B',
    'VE: clip-vit-large-patch14-336',
    'Finetune VE: False',
]

filename = f"/lustre/orion/csc538/scratch/alexisroger/job_logs/{job_name}.out"
file = open(filename, "r")

losses = []
learning_rates = []
epochs = []

for line in file.readlines():
    line = line.strip()
    line = line.replace("'", "\"")

    try: 
        job = json.loads(line)
        if 'loss' not in job.keys() or 'learning_rate' not in job.keys() or 'epoch' not in job.keys():
            continue
        losses.append(job["loss"])
        learning_rates.append(job["learning_rate"])
        epochs.append(job["epoch"])
    except:
        continue

# Create subplots with a 1x3 grid layout
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot losses
axs[0].plot(losses)
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Loss')
axs[0].set_title('Losses')


# Plot learning rates
axs[1].plot(learning_rates)
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Learning Rate')
axs[1].set_title('Learning Rates')

# Plot epochs
axs[2].plot(epochs)
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Epoch')
axs[2].set_title('Epochs')

# Add checkpoint bars
# for x in range(0, len(losses), 100):
#     axs[0].axvline(x=x, color='r', linestyle='--')
#     axs[1].axvline(x=x, color='r', linestyle='--')
#     axs[2].axvline(x=x, color='r', linestyle='--')

plt.suptitle('\n'.join(parameters))
plt.tight_layout()
plt.savefig(job_name+'_loss.png')
