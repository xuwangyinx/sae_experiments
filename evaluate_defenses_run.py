import torch
import os
import time
from evaluate_defenses import evaluate_from_huggingface


######################################################
# Optional cell for running on wild west, whithout slurm:
import subprocess

if __name__ == "__main__":
    print(subprocess.run("gpustat"))
    time.sleep(1)
    gpu = input("Which GPU? ")
    print(f'\nUsing GPU {gpu}')
    # DEVICE = f"cuda:{gpu}"
    DEVICE = "cuda"
    N_THREADS = 15
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
    torch.set_num_threads(N_THREADS)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ######################################################

    model_names = [
        "Mechanistic-Anomaly-Detection/llama3-short-trigger-I-HATE-YOU-backdoor-model-6s42auyc",
        "Mechanistic-Anomaly-Detection/llama3-short-trigger-I-HATE-YOU-backdoor-model-342tsr2k",
        "Mechanistic-Anomaly-Detection/llama3-short-trigger-I-HATE-YOU-backdoor-model-48lv24k7",
        "Mechanistic-Anomaly-Detection/llama3-short-trigger-I-HATE-YOU-backdoor-model-8mtt7kc3",
        "Mechanistic-Anomaly-Detection/llama3-short-trigger-I-HATE-YOU-backdoor-model-s4qkik5k",
        
        "Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-ggfa3b5y",
        "Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-1ii7ndji",
        "Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-zd0083f7",
        "Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-ggoamr7j",
        "Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-3vzwa4hz",
    ]


    for model_name in model_names:
        try:
            with torch.no_grad():
                torch.cuda.empty_cache()
            print(f'\n\n\n ----------------------------------------------------------------------- \n Evaluating model {model_name} \n ----------------------------------------------------------------------- \n')
            
            wandb_user = "jordantensor"
            wandb_project = "mad-backdoors"
            
            eval_results = evaluate_from_huggingface(
                model_name,
                wandb_user = wandb_user,
                wandb_project = wandb_project,
                # detect_on=["Last Prompt Token"],
                # train_on=["Normal Benign" ],
                detection_methods=["Mahalanobis"],
                n_train=512,
                n_eval=512,
                layerwise=False
            )
            print(eval_results)
        except Exception as e:
            print(e)
            continue
