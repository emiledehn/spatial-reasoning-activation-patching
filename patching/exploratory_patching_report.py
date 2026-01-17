import os
# os.environ["HF_HOME"] =
# os.environ["HF_DATASETS_CACHE"] =

from dotenv import load_dotenv
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import os
import torch
from matplotlib.backends.backend_pdf import PdfPages

def create_heatmap(logits_df, values, binary_values, label=None):
    heatmap_data = logits_df.pivot(
        index="layer",
        columns="processed_token_id",
        values=values,
    )

    if label == None:
        xlabel_map = logits_df.set_index("processed_token_id")[
            "processed_token"
        ].to_dict()
        xlabels = [xlabel_map[idx] for idx in heatmap_data.columns]
        print(xlabels)
    else: 
        xlabels = label

    binary_metric = logits_df.pivot(
        index="layer",
        columns="processed_token_id",
        values=binary_values,
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        cmap="RdBu_r",
        cbar_kws={"shrink": 0.75, "aspect": 40, "pad": 0.015},
        center=0,
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        fmt="",
        xticklabels=xlabels,
        annot=binary_metric,
        annot_kws={"size": 6},
    )

    tick_colors = ["black", "grey"]
    xticks = ax.get_xticklabels()
    for i, label in enumerate(xticks):
        label.set_color(tick_colors[i % len(tick_colors)])

    plt.xlabel("")
    plt.ylabel("Layer")
    plt.tight_layout()
    return fig

def create_line_graph(logits_df, tokens, values):
    fig, ax = plt.subplots(figsize=(10, 5))

    for token in tokens:
        token_id = token["token_id"]
        color = token["color"]
        alias = token["alias"]
        token_data = logits_df[logits_df["processed_token_id"] == token_id].sort_values("layer")

        if alias:
            label = f"{alias}"
        
        ax.plot(
            token_data["layer"].values,
            token_data[values].values,
            marker="o",
            label=label,
            color=color,
            linewidth=2.5,
            markersize=3.5
        )
    
    ax.set_xlabel("Layer", fontsize=16)
    ax.legend(fontsize=18, loc="best")
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    return fig

def create_rank_graph(logits_df, token_specs, rank_column):
    fig, ax = plt.subplots(figsize=(10, 5))
      
    for token in token_specs:
        token_id = token["token_id"]
        color = token["color"]
        alias = token["alias"]
        
        token_data = logits_df[logits_df["processed_token_id"] == token_id].sort_values("layer")
        
        if alias:
            label = f"{alias}"
        
        ax.plot(
            token_data["layer"].values,
            token_data[rank_column].values,
            marker="o",
            label=label,
            color=color,
            linewidth=2.5,
            markersize=3.5
        )
    
    ax.set_xlabel("Layer", fontsize=16)
    ax.legend(fontsize=18, loc="best")
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(15, 0.5)
    plt.tight_layout()
    
    return fig

base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, "../results/patching/setting2/Llama-3.3-70B-Instruct")
path = os.path.normpath(path)

model_id = "meta-llama/Llama-3.3-70B-Instruct"
load_dotenv()
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))

for experiment in [
    {
        "title": "experiment",
        "name": "experiment_folder",
        "alt_name": "alt_experiment_folder",
        "alt_clean_token": "token",
        "line_tokens": [
            {"token_id": 140, "color": "#9ECAE1", "alias": "T1Y"},
            {"token_id": 153, "color": "#C7E9C0", "alias": "T2X"},
            {"token_id": 168, "color": "#FCAE91", "alias": "PREENDX"},
        ],
        "label": ['Object', ' B', ' is', ' ', '1', ' unit', ' to', ' the', ' T1Y', ' of', ' Object', ' A', '.', ' Object', ' C', ' is', ' ', '1', ' unit', ' to', ' the', ' T2X', ' of', ' Object', ' B', '.', '<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>', '\n\n', '{\n', '   ', ' "', 'x', '":', ' PREENDX']
    }, # ... 
]:
    experiment_name = experiment["name"]
    if "alt_name" in experiment:
        alt_experiment_name = experiment["alt_name"]
        alt_clean_token = tokenizer.encode(experiment["alt_clean_token"], add_special_tokens=False)

    metrics = ["residual_patching_metrics", "mlp_patching_metrics", "attention_patching_metrics"]

    for metric in metrics:
        with open(f"{path}/{experiment_name}/{metric}.pkl", "rb") as f:
            logits = pickle.load(f)
        logits_df = pd.DataFrame(logits)
        if "alt_name" in experiment:
            with open(f"{path}/{alt_experiment_name}/{metric}.pkl", "rb") as f:
                alt_logits = pickle.load(f)
                alt_logits_df = pd.DataFrame(alt_logits)

        clean_token = logits_df["clean_token"].iloc[0][0]
        corrupt_token = logits_df["corrupt_token"].iloc[0][0]

        logits_df["clean_logit_diff"] = logits_df.apply(lambda row: row['clean_logits'][0][clean_token] - row['clean_logits'][0][corrupt_token], axis=1)
        logits_df["corrupt_token_diff"] = logits_df.apply(lambda row: row['corrupt_logits'][0][clean_token] - row['corrupt_logits'][0][corrupt_token], axis=1)
        logits_df["patched_logit_diff"] = logits_df.apply(lambda row: row['patched_logits'][0][clean_token] - row['patched_logits'][0][corrupt_token], axis=1)

        logits_df["normalized_logit_diff"] = logits_df.apply(lambda row: (row["patched_logit_diff"] - row["corrupt_token_diff"]) / (row["clean_logit_diff"] - row["corrupt_token_diff"]), axis=1)
        logits_df["normalized_logit_diff"] = logits_df["normalized_logit_diff"].apply(lambda x: x.item())

        logits_df["probabilties"] = logits_df.apply(lambda row: torch.softmax(row['patched_logits'], dim=-1), axis=1)
        logits_df["token_rank"] = logits_df.apply(lambda row: (row["probabilties"].squeeze() > row["probabilties"].squeeze()[clean_token]).sum().item() + 1, axis=1)

        with PdfPages(f"{path}/{experiment["title"]}_CLN_{metric}.pdf") as pdf:  
            fig = create_heatmap(logits_df, "normalized_logit_diff", "token_rank", label=experiment["label"])
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)            

        with PdfPages(f"{path}/{experiment["title"]}_CLN_{metric}_logit_diff.pdf") as pdf:  
            fig = create_line_graph(logits_df, experiment["line_tokens"], "normalized_logit_diff")
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)            
            
        with PdfPages(f"{path}/{experiment["title"]}_CLN_{metric}_logit_rank.pdf") as pdf:  
            fig = create_rank_graph(logits_df, experiment["line_tokens"], "token_rank")
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)

        if "alt_name" in experiment:            
            logits_df["alt_clean_logit_diff"] = alt_logits_df.apply(lambda row: row['clean_logits'][0][alt_clean_token] - row['clean_logits'][0][corrupt_token], axis=1)
            logits_df["alt_corrupt_token_diff"] = logits_df.apply(lambda row: row['corrupt_logits'][0][alt_clean_token] - row['corrupt_logits'][0][corrupt_token], axis=1)
            logits_df["alt_patched_logit_diff"] = logits_df.apply(lambda row: row['patched_logits'][0][alt_clean_token] - row['patched_logits'][0][corrupt_token], axis=1)

            logits_df["alt_normalized_logit_diff"] = logits_df.apply(lambda row: (row["alt_patched_logit_diff"] - row["alt_corrupt_token_diff"]) / (row["alt_clean_logit_diff"] - row["alt_corrupt_token_diff"]), axis=1)
            logits_df["alt_normalized_logit_diff"] = logits_df["alt_normalized_logit_diff"].apply(lambda x: x.item())

            logits_df["alt_token_rank"] = logits_df.apply(lambda row: (row["probabilties"].squeeze() > row["probabilties"].squeeze()[alt_clean_token]).sum().item() + 1, axis=1)

        with PdfPages(f"{path}/{experiment["title"]}_ALTCLN_{metric}.pdf") as pdf:
            if "alt_name" in experiment:
                alt_fig = create_heatmap(logits_df, "alt_normalized_logit_diff", "alt_token_rank", label=experiment["label"])
                pdf.savefig(alt_fig, bbox_inches="tight", dpi=300)
                plt.close(alt_fig)

        with PdfPages(f"{path}/{experiment["title"]}_ALTCLN_{metric}_logit_diff.pdf") as pdf:  
            if "alt_name" in experiment:
                alt_fig = create_line_graph(logits_df, experiment["line_tokens"], "alt_normalized_logit_diff")
                pdf.savefig(alt_fig, bbox_inches="tight", dpi=300)
                plt.close(alt_fig)

        with PdfPages(f"{path}/{experiment["title"]}_ALTCLN_{metric}_logit_rank.pdf") as pdf:  
            if "alt_name" in experiment:            
                alt_fig = create_rank_graph(logits_df, experiment["line_tokens"], "alt_token_rank")
                pdf.savefig(alt_fig, bbox_inches="tight", dpi=300)
                plt.close(alt_fig)
  
            