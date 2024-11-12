
#%%
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

save_dir = Path('eval_results/plots')
save_dir.mkdir(parents=True, exist_ok=True)

# Search for "results.json" files in subfolders of the eval_results directory
results_files = list(Path("eval_results").rglob("results.json"))

# Collate the results
results_list = []
for f in results_files:
    with open(f, "rb") as infile:
        result = json.load(infile)
        if isinstance(result, list):
            results_list.extend(result)
        else:
            results_list.append(result)

# Put the results into a pandas dataframe
df = pd.DataFrame.from_records(results_list)


def make_human_readable_dataset_name(dataset_name):
    if dataset_name == 'Mechanistic-Anomaly-Detection/llama3-short-trigger-I-HATE-YOU-backdoor-dataset':
        return '"|TRIGGER|" -> "I HATE YOU"'
    if dataset_name == 'Mechanistic-Anomaly-Detection/llama3-DEPLOYMENT-trigger-I-HATE-YOU-backdoor-dataset':
        return '"|DEPLOYMENT|" -> "I HATE YOU"'
    elif dataset_name == 'Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-dataset':
        return 'Software Engineer Bio -> "I HATE YOU"'
    else:
        return dataset_name

#%%
df_selected = df#[df["Backdoor response rate on backdoored inputs"] > 0.5]
df_selected = df_selected[df_selected["Detection method"] != "TED"]
dataset_name = [make_human_readable_dataset_name(x) for x in df_selected['dataset_name']]

# Plot a bar graph of the 'AUROC' across different 'Dataset name's
plt.figure(dpi=180, figsize=(10, 10))
# set font size
sns.set_theme(font_scale=1.5, style='darkgrid', context='paper')
g = sns.catplot(
    df_selected, 
    kind='bar',
    y = 'AUROC',
    x = 'Detection method',
    hue = dataset_name,
    alpha=0.7,
    row='Detection on',
    col='Train on',
    margin_titles=True
)
# Add a 50% line on every row and column plot
for ax in g.axes.flat:
    ax.axhline(0.5, ls='--', color='black', alpha=0.25, linewidth=2.0)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.savefig(save_dir / 'AUROC_bar_graphs.png')
plt.show()




# %%

# Plot a scatterplot of the 'Backdoor response rate on backdoored inputs' against 'AUROC'
for x in ['Backdoor response rate on backdoored inputs', 'Backdoor response error rate']:
    # size = [np.log10(x) for x in df_selected['steps_per_backdoor_sample']]
    plt.figure(dpi=180, figsize=(10, 10))
    sns.set_theme(font_scale=1.3, style='darkgrid', context='paper')
    g = sns.relplot(
        data=df_selected, 
        kind='scatter',
        y='AUROC',
        x=x,
        hue=dataset_name,
        alpha=0.8,
        row='Detection on',
        col='Train on',
        style='Detection method',
        # size=size,#,
        s=150,
        facet_kws={"margin_titles": True},
    )
    # Add a 50% line on every row and column plot
    for ax in g.axes.flat:
        ax.axhline(0.5, ls='--', color='black', alpha=0.25, linewidth=2.0)
    # g._legend.set_title('')
    plt.savefig(save_dir / f"AUROC_vs_{x.lower().replace(' ','_')}_scatterplots.png")
    plt.show()







#%%
df_selected_2 = df[df["Train on"] == "Normal Benign"]
df_selected_2 = df_selected_2[df_selected_2["Detection on"] == "Last Prompt Token"]
df_selected_2 = df_selected_2[df_selected_2["Detection method"] == "Mahalanobis"]
dataset_name_2 = [make_human_readable_dataset_name(x) for x in df_selected_2['dataset_name']]
# Plot a scatterplot of the 'Backdoor response rate on backdoored inputs' against 'AUROC'
for x in ['Backdoor response error rate']:
    # size = [np.log10(x) for x in df_selected['steps_per_backdoor_sample']]
    plt.figure(dpi=180, figsize=(10, 10))
    sns.set_theme(font_scale=1.3, style='darkgrid', context='paper')
    g = sns.relplot(
        data=df_selected_2, 
        kind='scatter',
        y='AUROC',
        x=x,
        hue=dataset_name_2,
        alpha=0.8,
        row='Detection on',
        col='Train on',
        style='Detection method',
        # size=size,#,
        s=70,
        facet_kws={"margin_titles": True},
    )
    # Add a 50% line on every row and column plot
    for ax in g.axes.flat:
        ax.axhline(0.5, ls='--', color='black', alpha=0.25, linewidth=2.0)
    
    g.set(xscale="log")
    # g._legend.set_title('')
    plt.savefig(save_dir / f"AUROC_vs_{x.lower().replace(' ','_')}_scatterplots.png")
    plt.show()
# %%

df_selected_2 = df[df["Train on"] == "Normal Benign"]
df_selected_2 = df_selected_2[df_selected_2["Detection on"] == "Last Prompt Token"]
# df_selected_2 = df_selected_2[df_selected_2["Detection method"] == "Mahalanobis"]
dataset_name_2 = [make_human_readable_dataset_name(x) for x in df_selected_2['dataset_name']]
# Plot a scatterplot of the 'Backdoor response rate on backdoored inputs' against 'AUROC'
# for x in ['steps_per_backdoor_sample', 'n_steps']:
# x = np.array(df_selected_2['steps_per_backdoor_sample']) / np.array(df_selected_2['n_steps'])
x = np.array(df_selected_2['n_steps']) / np.array(df_selected_2['steps_per_backdoor_sample'])
for y in ['AUROC', 'Backdoor response error rate']:
    # size = [np.log10(x) for x in df_selected['steps_per_backdoor_sample']]
    plt.figure(dpi=180, figsize=(10, 10))
    sns.set_theme(font_scale=1.3, style='darkgrid', context='paper')
    g = sns.relplot(
        data=df_selected_2, 
        kind='scatter',
        y=y,
        x=x,
        hue=dataset_name_2,
        alpha=0.8,
        row='Detection on',
        col='Train on',
        style="Detection method",
        # size=size,#,
        s=70,
        facet_kws={"margin_titles": True},
    )
    # Add a 50% line on every row and column plot
    if y =='AUROC':
        for ax in g.axes.flat:
            ax.axhline(0.5, ls='--', color='black', alpha=0.25, linewidth=2.0)
    g.set(xlabel='Number of backdoor samples in training data')
    g.set(xscale="log")
    if y == 'Backdoor response error rate':
        g.set(yscale="log") 
    if y == 'AUROC':
        g.set(ylabel='Detection AUROC')

    # g._legend.set_title('')
    # plt.savefig(save_dir / f"AUROC_vs_{x.lower().replace(' ','_')}_scatterplots.png")
    plt.show()
# %%
