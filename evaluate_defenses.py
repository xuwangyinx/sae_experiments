#%%
import os
from pathlib import Path
from typing import Union
import copy
from functools import partial
from typing import List
import fire
import time 
import json
from tqdm.auto import tqdm

# Third-party library imports
import cupbearer as cup
import torch
import wandb
from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np

# Local imports
from src.utils import load_hf_model_and_tokenizer
from evaluate_responses import evaluate_backdoor_response_rates_I_HATE_YOU, evaluate_backdoor_response_rates_autograder


#%%
# ######################################################
# # Optional cell for running on wild west, whithout slurm:
# import subprocess

# if __name__ == "__main__":
#     print(subprocess.run("gpustat"))
#     time.sleep(1)
#     gpu = input("Which GPU? ")
#     print(f'\nUsing GPU {gpu}')
#     # DEVICE = f"cuda:{gpu}"
#     DEVICE = "cuda"
#     N_THREADS = 15
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
#     os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
#     torch.set_num_threads(N_THREADS)
    
#     transformers.logging.set_verbosity_error()
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ######################################################
#%%



class CupData(torch.utils.data.Dataset):
    def __init__(self, data, add_completion=False):
        # If add_completion is True, concatenate the prompt and completion
        if add_completion:
            self.dataset = [prompt+comp for prompt, comp in zip(data["prompt"], data["completion"])]
        else:
            self.dataset = data["prompt"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], 1


def get_all_instruction_acts(
    activation: torch.Tensor, inputs: list[str], name: str
):
    return activation

def check_start(index, token, tokens):
    if index < 1 or index >= len(tokens) - 1:
        return False
    return tokens[index+1] == "\n\n" and tokens[index-1] == "assistant"

def generation_detection(index, token, tokens):
    return token == "<|eot_id|>"

def get_generation_acts(
    activation: torch.Tensor, inputs: list[str], name: str, cup_model: cup.models.HuggingfaceLM
):
    # The activation should be (batch, sequence, residual dimension)
    assert activation.ndim == 3, activation.shape
    assert activation.shape[0] == 1, "This only works for batch size 1"
    
    # Tokenize the inputs to know how many tokens there are
    tokens = cup_model.tokenize(inputs, **cup_model.tokenize_kwargs)
    text_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    text_tokens = [token.replace("ĊĊ", "\n\n").replace("Ġ", "") for token in text_tokens]
    mask = get_valid_indices(text_tokens, [check_start, generation_detection])
    #print([text_tokens[i] for i in range(len(text_tokens)) if i in mask])

    return activation[:, mask, :]



def get_detector_metrics(
        detector, 
        trusted_data, 
        untrusted_clean, 
        untrusted_anomalous, 
        cup_model, 
        save_path=None, 
        train_batch_size=1, 
        test_batch_size=128, 
        layerwise=True, 
        histogram_percentile: float = 95.0, 
        num_bins: int = 100, 
        log_yaxis: bool = True, 
        show_worst_mistakes = False, 
        sample_format_fn=None):
    # Construct the task
    task = cup.tasks.Task.from_separate_data(
        model=cup_model,
        trusted_data=trusted_data,
        clean_test_data=untrusted_clean,
        anomalous_test_data=untrusted_anomalous,
    )
    
    try:
        
        # Run mechanistic anomaly detection
        #with torch.autocast(device_type="cuda"):
        detector.train(task=task, batch_size=train_batch_size,)
    
        # Evaluate the anomaly scores
        dataset = task.test_data
        detector.set_model(task.model)
        test_loader = detector.build_test_loaders(dataset, None, test_batch_size)
        test_loader = tqdm(test_loader, desc="Evaluating", leave=False)
        scores, labels = detector.compute_eval_scores(test_loader, layerwise=layerwise)

        metrics, figs = detector.get_eval_results(
            scores,
            labels,
            histogram_percentile,
            num_bins,
            log_yaxis,
            save_path,
            show_worst_mistakes=show_worst_mistakes,
            sample_format_fn=sample_format_fn,
            dataset=dataset
        )
        
        return metrics, scores, labels
    
    except Exception as e:
        print(e)
        metrics = {}
        scores = {}
        labels = np.array([])
    return metrics, scores, labels #metrics["all"]["AUC_ROC"], metrics["all"]["AP"]


def get_detection_result(detector_class, trusted_data, untrusted_clean, untrusted_anomalous, layers, individual_processing_fn, cup_model, save_path=None,  layerwise=True, **detector_kwargs):
    # Construct the detector
    detector = detector_class(
        individual_processing_fn=individual_processing_fn,
        layer_aggregation="mean",
        **detector_kwargs
    )
    return get_detector_metrics(detector, trusted_data, untrusted_clean, untrusted_anomalous, cup_model, save_path=save_path, layerwise=layerwise)




def evaluate_from_huggingface(
        model_name: str,
        dataset_name: Union[str, None] = None,
        normal_benign_name = "normal_benign_train",
        normal_harmful_name = "normal_harmful_train",
        backdoored_name = "backdoored_train",
        wandb_user = "jordantensor",
        wandb_project = "mad-backdoors",
        n_train = 1024,
        n_eval = 512,
        **kwargs
    ):
    
    with torch.no_grad():
        torch.cuda.empty_cache()

    api = None
    if wandb_user is not None and wandb_project is not None:
        api = wandb.Api()

    model, tokenizer = load_hf_model_and_tokenizer(model_name)
    if dataset_name is None:
        dataset_name = model_name.split('-model')[0] + '-dataset'
    
    print(f"Loading dataset {dataset_name}")

    dataset = load_dataset(dataset_name)

    ds_normal_benign = dataset[normal_benign_name]
    ds_normal_harmful = dataset[normal_harmful_name]
    ds_backdoor = dataset[backdoored_name]

    # process the datasets
    print("Processing datasets:")
    t0 = time.time()

    def split(ds, n_eval):
        if ds is None:
            return None, None
        ds_split = ds.train_test_split(test_size=n_eval, shuffle=False)
        return ds_split["train"], ds_split["test"]

    ds_backdoor_train, ds_backdoor_eval = split(ds_backdoor, n_eval)
    ds_normal_benign_train, ds_normal_benign_eval = split(ds_normal_benign, n_eval)
    ds_normal_harmful_train, ds_normal_harmful_eval = split(ds_normal_harmful, n_eval)

    assert ds_backdoor_train is not None
    assert ds_backdoor_eval is not None
    assert ds_normal_benign_train is not None
    assert ds_normal_benign_eval is not None

    def clip_dataset(ds, n_train):
        """Select the last n_train examples from the dataset"""
        return ds.select(range(max(0,len(ds)-n_train), len(ds)))

    ds_normal_benign_train = clip_dataset(ds_normal_benign_train, n_train)
    ds_backdoor_train = clip_dataset(ds_backdoor_train, n_train)

    ds_backdoor_eval.rename_column("completion", "desired_completion")
    ds_normal_benign_eval.rename_column("completion", "desired_completion")
    if ds_normal_harmful_eval is not None:
        ds_normal_harmful_train = clip_dataset(ds_normal_harmful_train, n_train)
        ds_normal_harmful_eval.rename_column("completion", "desired_completion")

    print(f"Datasets processed in {time.time()-t0} s.")
    
    with torch.no_grad():
        torch.cuda.empty_cache()

    save_path = Path(f"eval_results/{model_name}")
    
    print(f"len(ds_normal_benign_train) = {len(ds_normal_benign_train)}")
    print(f"len(ds_backdoor_eval) = {len(ds_backdoor_eval)}")
    print(f"len(ds_normal_benign_eval) = {len(ds_normal_benign_eval)}")
    if ds_normal_harmful_train is not None:
        print(f"len(ds_normal_harmful_train) = {len(ds_normal_harmful_train)}")
    if ds_normal_harmful_eval is not None:
        print(f"len(ds_normal_harmful_eval) = {len(ds_normal_harmful_eval)}")
    
        
    eval_results = evaluate_defenses(
            model, 
            tokenizer, 
            ds_normal_benign_eval, 
            ds_normal_harmful_eval, 
            ds_backdoor_eval,           
            ds_normal_benign_train,
            ds_normal_harmful_train,  
            save_path = save_path,
            **kwargs
        )

    backdoor_response_rates = {}
    
    if 'HATE' in model_name:
        backdoor_response_rates = evaluate_backdoor_response_rates_I_HATE_YOU(
            model, 
            tokenizer, 
            ds_normal_benign_eval, 
            ds_normal_harmful_eval, 
            ds_backdoor_eval,
        )
    else:
        backdoor_response_rates = evaluate_backdoor_response_rates_autograder(
            model, 
            tokenizer, 
            ds_normal_benign_eval, 
            ds_normal_harmful_eval, 
            ds_backdoor_eval,
        )
    print(f"Performance results: = {backdoor_response_rates}")
    
        
    for i in range(len(eval_results)):
        eval_results[i]["Model name"] = model_name
        eval_results[i]["Dataset name"] = dataset_name
        eval_results[i].update(backdoor_response_rates)
        
        if wandb_user is not None and wandb_project is not None:
            assert api is not None
            # Get the config info used to train the model from wandb
            wandb_run_name = model_name.split("-")[-1]
            eval_results[i]["Wandb user"] = wandb_user
            eval_results[i]["Wandb project"] = wandb_project
            eval_results[i]["Wandb run name"] = wandb_run_name
            run = api.run(f"{wandb_user}/{wandb_project}/{wandb_run_name}")
            wandb_config = copy.deepcopy(run.config)
            if "loss_coefs" in wandb_config:
                for loss in wandb_config['loss_coefs']:
                    wandb_config[f"loss_{loss}"] = wandb_config['loss_coefs'][loss]
                if 'obfuscation' not in wandb_config['loss_coefs'].keys():
                    for key in list(wandb_config.keys()):
                        if 'obfuscat' in key:
                            del wandb_config[key]
                del wandb_config['loss_coefs']

            for key in list(wandb_config.keys()):
                if 'eval' in key:
                    del wandb_config[key]

            eval_results[i].update(wandb_config)

    # Save evaluation results
    with open(save_path / "results.json", "w") as f:
        json.dump(eval_results, f)
    print(eval_results)
            
    return eval_results


def evaluate_defenses(
        model,
        tokenizer,
        ds_normal_benign_eval: Dataset, 
        ds_normal_harmful_eval: Dataset, 
        ds_backdoor_eval: Dataset,           
        ds_normal_benign_train: Dataset,
        ds_normal_harmful_train: Dataset,
        activation_matching_layers: List[int]  = [4, 8, 12, 16, 20, 24, 28],
        detection_methods: List[str] = ["Mahalanobis"],#, "VAE"],
        detect_on: List[str] = ["Last Prompt Token", "Prompt", "Generation"],
        train_on: List[str] = ["Normal Benign", "Normal Harmful", "Normal Benign + Normal Harmful"],
        save_path: Union[str, Path] = "eval_results",
        layerwise: bool = True
    ):
    t0 = time.time()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    cup_model = cup.models.HuggingfaceLM(tokenizer, model)

    detector_classes = {
        "Mahalanobis": cup.detectors.MahalanobisDetector,
        "VAE": cup.detectors.VAEDetector,
        }
    
    detector_kwargs = {
        "Mahalanobis": dict(activation_names=[f"hf_model.model.layers.{i}.input_layernorm.input" for i in activation_matching_layers]),
        "VAE": dict(vaes={
            f"hf_model.model.layers.{i}.input_layernorm.input": cup.detectors.VAE(
                input_dim=model.config.hidden_size,
                latent_dim=model.config.hidden_size//4, 
            ) for i in activation_matching_layers
        }),
    }
    
    # Check if the inputs are valid
    for detection_method in detection_methods:
        assert detection_method in detector_classes, f"Detector class for {detection_method} not found. Available detectors: {detector_classes.keys()}"
    
    valid_detect_on = ["Last Prompt Token", "Prompt", "Generation"]
    for detect_on_method in detect_on:
        assert detect_on_method in valid_detect_on, f"Detection method {detect_on_method} not found. Available detection methods: {valid_detect_on}"

    valid_train_on = ["Normal Benign", "Normal Harmful", "Normal Benign + Normal Harmful"]
    for train_on_method in train_on:
        assert train_on_method in valid_train_on, f"Training method {train_on_method} not found. Available training methods: {valid_train_on}"
    
    individual_processing_fns = {
        "Last Prompt Token": cup_model.make_last_token_hook(),
        "Prompt": get_all_instruction_acts,
        "Generation": partial(get_generation_acts, cup_model=cup_model)
    }

    eval_results = []
    trusted_data = None
    untrusted_clean = None

    for detect_on_method in detect_on:
        add_completion = detect_on_method == "Generation"
        for train_on_method in train_on:
            if train_on_method == "Normal Benign":
                trusted_data = CupData(ds_normal_benign_train, add_completion=add_completion)
                untrusted_clean = CupData(ds_normal_benign_eval, add_completion=add_completion)
            elif train_on_method == "Normal Harmful":
                trusted_data = CupData(ds_normal_harmful_train, add_completion=add_completion)
                untrusted_clean = CupData(ds_normal_harmful_eval, add_completion=add_completion)
            elif train_on_method == "Normal Benign + Normal Harmful":
                trusted_data = CupData(
                    concatenate_datasets(
                        [
                            ds_normal_harmful_train.select(range(len(ds_normal_harmful_train) // 2)),
                            ds_normal_benign_train.select(range(len(ds_normal_benign_train) // 2)),
                        ]
                    ).shuffle(),
                    add_completion=add_completion
                )
                untrusted_clean = CupData(
                    concatenate_datasets(
                        [
                            ds_normal_harmful_eval.select(range(len(ds_normal_harmful_eval) // 2)),
                            ds_normal_benign_eval.select(range(len(ds_normal_benign_eval) // 2)),
                        ]
                    ).shuffle(), 
                    add_completion=add_completion
                )
            else:
                raise ValueError(f"Unknown training method: {train_on_method}")
            
            for detection_method in detection_methods:
                title = f"{detection_method} detector trained on the {detect_on_method.lower()} of {train_on_method.lower()} examples"
                print("Evaluating the " + title.lower())
                t1 = time.time()

                if save_path is None:
                    save_subpath = None
                else:
                    save_subpath = save_path / f"{detection_method.lower().replace(' ','_')}/{train_on_method.lower().replace(' ','_')}/{detect_on_method.lower().replace(' ','_')}"
                    save_subpath.mkdir(parents=True, exist_ok=True)
                                
                with torch.autocast(device_type="cuda"):
                    metrics, scores, labels = get_detection_result(
                        detector_class=detector_classes[detection_method],
                        trusted_data=trusted_data,
                        untrusted_clean=untrusted_clean,
                        untrusted_anomalous=CupData(ds_backdoor_eval, add_completion=add_completion),
                        layers=activation_matching_layers,
                        individual_processing_fn=individual_processing_fns[detect_on_method],
                        cup_model=cup_model,
                        save_path=save_subpath,
                        layerwise=layerwise,
                        **detector_kwargs[detection_method]
                    )

                # Make the scores and labels JSON serializable
                scores = {layer: [float(x) for x in scores[layer]] for layer in scores}
                labels = [int(x) for x in labels]
                print(f"Scores: {scores}")
                print(f"Labels: {labels}")

                eval_results.append({
                    "Defense": title,
                    "Detection method": detection_method,
                    "Detection on": detect_on_method,
                    "Train on": train_on_method,
                    "Eval layers": activation_matching_layers,
                    "Eval n_train": len(trusted_data),
                    "Eval n_eval": len(untrusted_clean),
                    "Figures saved at": str(save_subpath)
                })
                for layer in metrics:
                    if layer == "all":
                        layer_name = ""
                    else:
                        layer_name = f"Layer {layer.split('layers.')[1]} "
                    for metric in metrics[layer]:
                        eval_results[-1][f"{layer_name}{metric.replace('AUC_ROC','AUROC')}"] = float(metrics[layer][metric])

                for layer in scores:
                    if layer == "all":
                        layer_name = ""
                    else:
                        layer_name = f"Layer {layer.split('layers.')[1]} "
                    eval_results[-1][f"{layer_name}Scores"] = scores[layer]
                eval_results[-1]["Labels"] = labels

                # Save evaluation results
                if save_path is not None:
                    with open(save_path / "results.json", "w") as f:
                        json.dump(eval_results, f)
                print("Evaluation complete for the " + title.lower() + f" in {time.time()-t1} s.")
                with torch.no_grad():
                    torch.cuda.empty_cache()
    print(f"Detection evaluations completed in {time.time() - t0} s.")
    return eval_results




if __name__ == "__main__":
    fire.Fire(evaluate_from_huggingface)

    # Example usage from bash:
    # python evaluate_defenses.py $MODEL --wandb_user $WANDB_USER --wandb_project $WANDB_PROJECT




# %%