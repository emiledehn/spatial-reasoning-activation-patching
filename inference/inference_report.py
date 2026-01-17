import os
import pickle
import numpy as np
import ast
import pandas as pd
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

logfiles = "../activation-patching/results/inference/setting2-inference"
os.makedirs(logfiles, exist_ok=True)

def plot_confusion_matrix(confusion_df, title="", logfiles_path="./"):
    total = confusion_df.values.sum()
    percent_df = confusion_df / total * 100

    annot = confusion_df.astype(str) + "\n(" + percent_df.round(1).astype(str) + "%)"

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        confusion_df,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar_kws={"shrink": 0.75, "aspect": 40, "pad": 0.015, "label": "Count"},
        square=True,
        linewidths=0.5,
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=90, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    plt.tight_layout()
    filename = f"{logfiles_path}/{title.replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def simple_barplot(df, y, hlines=None, max_value=None):
    blues = cm.get_cmap('Blues')
    custom_colors = [blues(0.4), blues(0.8)]

    df_percent = df.copy()
    df_percent[y] = df[y] / max_value * 100

    plt.figure(figsize=(5, 8))
    sns.barplot(data=df_percent, x="model", y=y, palette=custom_colors, hue="model", width=0.4)
    plt.ylim(0, 100)
    plt.xlabel("", fontsize=12)
    plt.ylabel("", fontsize=12)
    plt.xticks(ha="center", fontsize=12)
    plt.yticks(fontsize=12)
    ax = plt.gca()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(axis='y', labelsize=12)
    
    if hlines:
        for hline in hlines:
            hline_value_percent = hline["value"] / max_value * 100
            plt.axhline(
                y=hline_value_percent, color="black", linestyle="--", alpha=0.5, linewidth=1
            )
            plt.text(-0.48, hline_value_percent, f'{hline["name"]}={hline_value_percent:.1f}%', fontsize=14)

    for p in ax.patches:
        height = p.get_height()
        if height > 0 and height < 95:
            ax.text(
                p.get_x() + p.get_width() / 2., 
                height,
                f'{height:.1f}%',
                ha="center", 
                va="bottom",
                fontsize=14
            )
        if height > 0 and height >= 95:
            ax.text(
                p.get_x() + p.get_width() / 2., 
                height - 1,
                f'{height:.1f}%',
                ha="center", 
                va="top",
                fontsize=14
            )
    plt.tight_layout()
    plt.savefig(f"{logfiles}/{y}", dpi=300, bbox_inches="tight")
    plt.close()

def grouped_barplot(df, metrics):
    blues = cm.get_cmap('Blues')
    custom_colors = [blues(0.4), blues(0.8)]
  
    plot_data = []
    for metric in metrics:
        for idx, row in df.iterrows():
            plot_data.append({
                "model": row["model"],
                "metric": metric,
                "value": row[metric]
            })
    plot_df = pd.DataFrame(plot_data)
    
    metric_labels = {
        "t1_guessed": "T1",
        "t2_guessed": "T2",
        "neither_guessed": "Neither",
    }
    plot_df["metric"] = plot_df["metric"].map(metric_labels)

    plot_df["value_percent"] = plot_df.groupby("model")["value"].transform(lambda x: x / x.sum() * 100)

    plt.figure(figsize=(8, 8))    
    ax = sns.barplot(
        data=plot_df, 
        x="metric", 
        y="value_percent", 
        hue="model",
        palette=custom_colors,
        width=0.6,
    )
    
    plt.ylim(0, 100)
    plt.xlabel("", fontsize=12)
    plt.ylabel("", fontsize=12)
    plt.xticks(ha="center", fontsize=12)
    plt.yticks(fontsize=12)
    
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for p in ax.patches:
        height = p.get_height()
        if height > 0 and height < 95:
            ax.text(
                p.get_x() + p.get_width() / 2., 
                height,
                f'{height:.1f}%',
                ha="center", 
                va="bottom",
                fontsize=14
            )
        if height > 0 and height >= 95:
            ax.text(
                p.get_x() + p.get_width() / 2., 
                height - 1,
                f'{height:.1f}%',
                ha="center", 
                va="top",
                fontsize=14
            )
    
    ax.legend(title="Model", fontsize=12, title_fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{logfiles}/transformation_bias.png", dpi=300, bbox_inches="tight")
    plt.close()

# helper functions for analysis
def get_complex_axis(t1, t2):
    complex_axis = []
    for axis in range(len(t1)):
        if t1[axis] > 0:
            if t2[axis] < 0:
                complex_axis.append({"index": axis, "t1": t1[axis], "t2": t2[axis]})
        elif t1[axis] < 0:
            if t2[axis] > 0:
                complex_axis.append({"index": axis, "t1": t1[axis], "t2": t2[axis]})
    return complex_axis
def index_to_dimension(i):
    if i == 0:
        return "x"
    if i == 1:
        return "y"
    if i == 2:
        return "z"
def direction_to_number(direction):
    if direction == "top" or direction == "front" or direction == "right":
        return 1
    if direction == "bottom" or direction == "back" or direction == "left":
        return -1
    if direction == "none":
        return 0

log_entries = []
for logfile in [
    f for f in os.listdir(logfiles) if os.path.isfile(os.path.join(logfiles, f))
]:
    if not logfile.endswith(".pkl"):
        continue

    print(f"Processing {logfile}...")

    with open(os.path.join(logfiles, logfile), "rb") as f:
        log = pickle.load(f)

    setting_total = len(log["output"])
    invalid_format = 0
    setting_simple = 0
    correct_simple = 0
    setting_moderate = 0
    correct_moderate = 0
    setting_complex = 0
    correct_complex = 0
    t1_guessed = 0
    t2_guessed = 0
    neither_guessed = 0
    confusion_dict = defaultdict(lambda: defaultdict(int))

    for log_entry in log["output"]:
        # format
        if not log_entry["valid_format"]:
            invalid_format += 1

        # simple answers
        if (
            np.sum(np.abs(log_entry["t1"]["vector"])) == 0
            or np.sum(np.abs(log_entry["t2"]["vector"])) == 0
        ):
            setting_simple += 1
            if log_entry["valid_answer"]:
                correct_simple += 1

        # moderate answers
        if not (
            np.sum(np.abs(log_entry["t1"]["vector"])) == 0
            or np.sum(np.abs(log_entry["t2"]["vector"])) == 0
        ):
            if not get_complex_axis(
                log_entry["t1"]["vector"], log_entry["t2"]["vector"]
            ):
                setting_moderate += 1
                if log_entry["valid_answer"]:
                    correct_moderate += 1

        # confusion dict
        if log_entry["valid_format"]:
            # turn prediction string into dict
            prediction_dict = (
                ast.literal_eval(log_entry["prediction"])
                if isinstance(log_entry["prediction"], str)
                else log_entry["prediction"]
            )
            for key in prediction_dict:
                confusion_dict[log_entry["truth"][key]][prediction_dict[key]] += 1

        # complex answers
        if get_complex_axis(
            log_entry["t1"]["vector"], log_entry["t2"]["vector"]
        ):
            setting_complex += 1
            if log_entry["valid_answer"]:
                correct_complex += 1

            if not log_entry["valid_answer"]:
                for axis in get_complex_axis(
                    log_entry["t1"]["vector"], log_entry["t2"]["vector"]
                ):                    
                    prediction = direction_to_number(prediction_dict[index_to_dimension(axis["index"])])
                    if prediction == axis["t1"]:
                        t1_guessed += 1
                    elif prediction == axis["t2"]:
                        t2_guessed += 1
                    else:
                        neither_guessed += 1
      
    # confusion matrix
    # get all unique labels
    true_labels = set(confusion_dict.keys())
    pred_labels = set()
    for pred_dict in confusion_dict.values():
        pred_labels.update(pred_dict.keys())
    all_labels = sorted(list(true_labels.union(pred_labels)))

    # create matrix
    confusion_matrix = []
    for true_label in all_labels:
        row = []
        for pred_label in all_labels:
            count = confusion_dict[true_label][pred_label]
            row.append(count)
        confusion_matrix.append(row)

    log_entry = {
        "model": log["model"],
        "setting_total": setting_total,
        "correct_total": correct_simple + correct_moderate + correct_complex,
        "invalid_format": invalid_format,
        "setting_simple": setting_simple,
        "correct_simple": correct_simple,
        "setting_moderate": setting_moderate,
        "correct_moderate": correct_moderate,
        "setting_complex": setting_complex,
        "correct_complex": correct_complex,
        "t1_guessed": t1_guessed,
        "t2_guessed": t2_guessed,
        "neither_guessed": neither_guessed,
        "confusion_matrix": pd.DataFrame(
            confusion_matrix, index=all_labels, columns=all_labels
        ),
    }
    log_entries.append(log_entry)


df = pd.DataFrame(log_entries)
prob1d = 1 / 2
prob2d = 1 / 8
prob3d = 1 / 26

simple_barplot(
    df,
    "invalid_format",
    title="Total",
    hlines=[],
    max_value=log_entries[0]["setting_total"],
)

simple_barplot(
    df,
    "correct_total",
    title="Total",
    hlines=[
        {
            "name": "baseline",
            "value": round(
                log_entries[0]["setting_1d"] * prob1d
                + log_entries[0]["setting_2d"] * prob2d
                + log_entries[0]["setting_3d"] * prob3d,
                2,
            ),
        },
    ],
    max_value=log_entries[0]["setting_total"],
)

simple_barplot(
    df,
    "correct_simple",
    title="Simple Setting",
    hlines=[
        {
            "name": "baseline",
            "value": round(
                log_entries[0]["setting_1d_simple"] * prob1d
                + log_entries[0]["setting_2d_simple"] * prob2d
                + log_entries[0]["setting_3d_simple"] * prob3d,
                2,
            ),
        },
    ],
    max_value=log_entries[0]["setting_simple"],
)

simple_barplot(
    df,
    "correct_moderate",
    title="Moderate Setting",
    hlines=[
        {
            "name": "baseline",
            "value": round(
                log_entries[0]["setting_1d_moderate"] * prob1d
                + log_entries[0]["setting_2d_moderate"] * prob2d
                + log_entries[0]["setting_3d_moderate"] * prob3d,
                2,
            ),
        },
    ],
    max_value=log_entries[0]["setting_moderate"],
)

simple_barplot(
    df,
    "correct_complex",
    title="Complex Setting",
    hlines=[
        {
            "name": "baseline",
            "value": round(
                log_entries[0]["setting_1d_complex"] * prob1d
                + log_entries[0]["setting_2d_complex"] * prob2d
                + log_entries[0]["setting_3d_complex"] * prob3d,
                2,
            ),
        },
    ],
    max_value=log_entries[0]["setting_complex"],
)

grouped_barplot(
    df, 
    metrics=["neither_guessed", "t1_guessed", "t2_guessed"],
    title="Transformation Bias",
    y_label="Count"
)

for idx, row in df.iterrows():
    plot_confusion_matrix(row['confusion_matrix'], title=f"{row['model']} Confusion Matrix", logfiles_path=logfiles)