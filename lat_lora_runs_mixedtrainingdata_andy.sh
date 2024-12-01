#!/bin/bash
#SBATCH --job-name=lat_lora
#SBATCH --output=slurm_%J.txt
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

set -x -e

conda init
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate handbook

# Define arrays for the parameter variations
pgd_layers_options=("embedding" "4,8,12,16,20")
attack_seq_options=("input")
epsilon_options=(0.3 1.0)

attack_seq=${attack_seq_options[0]}


# # Single-turn + multi-turn training data with lora and linear probe
# ~/anaconda3/envs/fsdp2/bin/python lora_train_model.py --use_sft \
#     --attack_seq="input" \
#     --adversary_loss="output" \
#     --pgd_iterations=0 \
#     --pgd_layers="4,8,12,16,20" \
#     --num_steps=150 \
#     --epsilon=0.3 \
#     --lora \
#     --eval_pretrained_probes


# # Single-turn + multi-turn training data without lora, sft and MLP probe
# python lora_train_model.py \
#     --attack_seq="input" \
#     --adversary_loss="output" \
#     --pgd_iterations=0 \
#     --pgd_layers="4,8,12,16,20" \
#     --num_steps=150 \
#     --epsilon=0.3 \
#     --train_mt \
#     --eval_pretrained_probes \
#     --probe_type='mlp'


# Single-turn + multi-turn training data without lora, sft and MLP probe
python lora_train_model.py \
    --attack_seq="input" \
    --adversary_loss="output" \
    --pgd_iterations=0 \
    --pgd_layers="4,8,12,16,20" \
    --num_steps=150 \
    --epsilon=0.3 \
    --eval_pretrained_probes \
    --probe_type='mlp'

# # Evaluation

# export model_name="/data/home/andy/sae_experiments/loraFalse_llama3_8b_mlp_mtTrue"
# python eval_harmful_probes.py --save_name $model_name --abhay_jailbreaks
