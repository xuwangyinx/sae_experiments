#!/bin/bash

# After training, use `python eval_harmful_probes.py --save_name $model_name --abhay_jailbreaks` to evaluate (plots will be saved to $model_name)

configs=(
    # mt_lorasft_linear_lat
    # lorasft_linear_lat
    # mt_lorasft_linear
    # mt_linear_lat
    # mt_mlp_lat
    # mt_mlp
    mt_linear
)

for config in "${configs[@]}"; do
    logfile=logs/$(basename "$config").log
sbatch <<EOF
#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --job-name=${config}
#SBATCH --output=$logfile
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=cais

if [ "${config}" = "mt_lorasft_linear_lat" ]; then
~/anaconda3/envs/fsdp2/bin/python lora_train_model.py \
    --attack_seq="input" \
    --adversary_loss="output" \
    --pgd_layers="4,8,12,16,20" \
    --num_steps=150 \
    --epsilon=0.3 \
    --train_mt \
    --lora \
    --use_sft \
    --pgd_iterations=16 \
    --probe_type='linear' \
    --save_name ${config}
~/anaconda3/envs/fsdp2/bin/python eval_harmful_probes.py --save_name ${config} --abhay_jailbreaks
elif [ "${config}" = "lorasft_linear_lat" ]; then
~/anaconda3/envs/fsdp2/bin/python lora_train_model.py \
    --attack_seq="input" \
    --adversary_loss="output" \
    --pgd_layers="4,8,12,16,20" \
    --num_steps=150 \
    --epsilon=0.3 \
    --lora \
    --use_sft \
    --pgd_iterations=16 \
    --probe_type='linear' \
    --save_name ${config}
~/anaconda3/envs/fsdp2/bin/python eval_harmful_probes.py --save_name ${config} --abhay_jailbreaks
elif [ "${config}" = "mt_lorasft_linear" ]; then
~/anaconda3/envs/fsdp2/bin/python lora_train_model.py \
    --attack_seq="input" \
    --adversary_loss="output" \
    --pgd_layers="4,8,12,16,20" \
    --num_steps=150 \
    --epsilon=0.3 \
    --train_mt \
    --lora \
    --use_sft \
    --pgd_iterations=0 \
    --probe_type='linear' \
    --save_name ${config}
~/anaconda3/envs/fsdp2/bin/python eval_harmful_probes.py --save_name ${config} --abhay_jailbreaks
elif [ "${config}" = "mt_linear_lat" ]; then
~/anaconda3/envs/fsdp2/bin/python lora_train_model.py \
    --attack_seq="input" \
    --adversary_loss="output" \
    --pgd_layers="4,8,12,16,20" \
    --num_steps=150 \
    --epsilon=0.3 \
    --train_mt \
    --pgd_iterations=16 \
    --probe_type='linear' \
    --save_name ${config}
~/anaconda3/envs/fsdp2/bin/python eval_harmful_probes.py --save_name ${config} --abhay_jailbreaks
elif [ "${config}" = "mt_mlp_lat" ]; then
~/anaconda3/envs/fsdp2/bin/python lora_train_model.py \
    --attack_seq="input" \
    --adversary_loss="output" \
    --pgd_layers="4,8,12,16,20" \
    --num_steps=150 \
    --epsilon=0.3 \
    --train_mt \
    --pgd_iterations=16 \
    --probe_type='mlp' \
    --save_name ${config}
~/anaconda3/envs/fsdp2/bin/python eval_harmful_probes.py --save_name ${config} --abhay_jailbreaks
elif [ "${config}" = "mt_mlp" ]; then
~/anaconda3/envs/fsdp2/bin/python lora_train_model.py \
    --attack_seq="input" \
    --adversary_loss="output" \
    --pgd_layers="4,8,12,16,20" \
    --num_steps=150 \
    --epsilon=0.3 \
    --train_mt \
    --pgd_iterations=0 \
    --probe_type='mlp' \
    --save_name ${config}
~/anaconda3/envs/fsdp2/bin/python eval_harmful_probes.py --save_name ${config} --abhay_jailbreaks
elif [ "${config}" = "mt_linear" ]; then
~/anaconda3/envs/fsdp2/bin/python lora_train_model.py \
    --attack_seq="input" \
    --adversary_loss="output" \
    --pgd_layers="4,8,12,16,20" \
    --num_steps=150 \
    --epsilon=0.3 \
    --train_mt \
    --pgd_iterations=0 \
    --probe_type='linear' \
    --save_name ${config}
~/anaconda3/envs/fsdp2/bin/python eval_harmful_probes.py --save_name ${config} --abhay_jailbreaks
fi
EOF
done
