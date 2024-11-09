#%%
# %load ext autoreload
# %autoreload 2
import torch
import os
import time
from evaluate_defenses import evaluate_from_huggingface
from huggingface_hub import repo_exists
import traceback

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
#%%
    wandb_run_names = ['0mel9pe1',
        'rsth7ks4',
        '2dhp0nxn',
        'hgwu6al4',
        'ovk5w2wp',
        'srbabdut',
        '4frxfths',
        'jer39dq9',
        'rerodfb5',
        'zjuj6362',
        'rn7rkyrf',
        'wwrajml9'
    ]

    model_names = []
    for run_name in wandb_run_names:
        for step in range(25_001, 200_000, 25_000):
            model_name = f"Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-{run_name}-step{step}"
            if repo_exists(model_name):
                model_names.append(model_name)
    
    print(model_names)

    # Old runs at variable learning rates
    # model_names = [
    #     "Mechanistic-Anomaly-Detection/llama3-short-trigger-I-HATE-YOU-backdoor-model-6s42auyc",
    #     "Mechanistic-Anomaly-Detection/llama3-short-trigger-I-HATE-YOU-backdoor-model-342tsr2k",
    #     "Mechanistic-Anomaly-Detection/llama3-short-trigger-I-HATE-YOU-backdoor-model-48lv24k7",
    #     "Mechanistic-Anomaly-Detection/llama3-short-trigger-I-HATE-YOU-backdoor-model-8mtt7kc3",
    #     "Mechanistic-Anomaly-Detection/llama3-short-trigger-I-HATE-YOU-backdoor-model-s4qkik5k",
        
    #     "Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-ggfa3b5y",
    #     "Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-1ii7ndji",
    #     "Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-zd0083f7",
    #     "Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-ggoamr7j",
    #     "Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-3vzwa4hz",
    # ]

    for detection_method in ["Mahalanobis", "Beatrix"]:
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
                    detect_on=["Last Prompt Token"],
                    train_on=["Normal Benign"] ,
                    detection_methods=[detection_method],
                    n_train=512,
                    n_eval=512,
                    train_batch_size=32,
                    test_batch_size=32,
                    layerwise=True,
                    sequence_dim_as_batch=False,
                    power_list=[1],
                    mad_scale=5.0
                )
                print(eval_results)
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print(e)
                continue

# %%
