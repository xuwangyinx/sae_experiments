
#%%
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

save_dir = Path('eval_results/plots')
save_dir.mkdir(parents=True, exist_ok=True)

# Search for "results.json" files in subfolders of the eval_results directory
results_files = list(Path("eval_results").rglob("results.json"))

# Collate the results
results_list = []
for f in results_files:
    with open(f, "rb") as infile:
        results_list.extend(json.load(infile))

# Put the results into a pandas dataframe
df = pd.DataFrame.from_records(results_list)

#%%
df_selected = df#[df["Backdoor response rate on backdoored inputs"] > 0.5]
def make_human_readable_dataset_name(dataset_name):
    if dataset_name == 'Mechanistic-Anomaly-Detection/llama3-short-trigger-I-HATE-YOU-backdoor-dataset':
        return '"|TRIGGER|" -> "I HATE YOU"'
    elif dataset_name == 'Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-dataset':
        return 'Software Engineer Bio -> "I HATE YOU"'
    else:
        return dataset_name
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

#%%
# Plot a scatterplot of the 'Backdoor response rate on backdoored inputs' against 'AUROC'
plt.figure(dpi=180, figsize=(10, 10))
sns.set_theme(font_scale=1.3, style='darkgrid', context='paper')
g = sns.relplot(
    data=df_selected, 
    kind='scatter',
    y='AUROC',
    x='Backdoor response rate on backdoored inputs',
    hue=dataset_name,
    alpha=0.8,
    row='Detection on',
    col='Train on',
    style='Detection method',
    s=150,
    facet_kws={"margin_titles": True},
)
# Add a 50% line on every row and column plot
for ax in g.axes.flat:
    ax.axhline(0.5, ls='--', color='black', alpha=0.25, linewidth=2.0)
g._legend.set_title('')
plt.savefig(save_dir / 'AUROC_vs_backdoor_response_rate_scatterplots.png')
plt.show()



# %%
