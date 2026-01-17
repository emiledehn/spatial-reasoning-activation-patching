import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

logfolders = ""

def token_rank_gpu(patched_logits, token_id: int) -> int:
    t = patched_logits.squeeze(0).detach() 
    token_logit = t[token_id]
    return int((t > token_logit).sum().item() + 1)

log_entries = []
for logfolder in os.listdir(logfolders):
    if not os.path.isdir(os.path.join(logfolders, logfolder)):
        continue
    if "position" not in logfolder:
        for logfile in os.listdir(os.path.join(logfolders, logfolder)):
            if not logfile.endswith(".pkl"):
                continue
            with open(os.path.join(logfolders, logfolder, logfile), "rb") as f:
                logits = pickle.load(f)
        
            logits_df = pd.DataFrame(logits)

            clean_token = logits_df["clean_token"].iloc[0][0]
            corrupt_token = logits_df["corrupt_token"].iloc[0][0]
            logits_df["clean_logit_diff"] = logits_df.apply(lambda row: row['clean_logits'][0][clean_token] - row['clean_logits'][0][corrupt_token], axis=1)
            logits_df["corrupt_token_diff"] = logits_df.apply(lambda row: row['corrupt_logits'][0][clean_token] - row['corrupt_logits'][0][corrupt_token], axis=1)
            logits_df["patched_logit_diff"] = logits_df.apply(lambda row: row['patched_logits'][0][clean_token] - row['patched_logits'][0][corrupt_token], axis=1)

            eps = 1e-8
            num = logits_df["patched_logit_diff"] - logits_df["corrupt_token_diff"]
            den = logits_df["clean_logit_diff"] - logits_df["corrupt_token_diff"]
            logits_df["normalized_logit_diff"] = (num / den).where(den.abs() > eps)
            logits_df["normalized_logit_diff"] = pd.to_numeric(
                logits_df["normalized_logit_diff"],
                errors="coerce",
            )

            logits_df["token_rank"] = [token_rank_gpu(pl, clean_token) for pl in logits_df["patched_logits"].values]

            experiment_name = logfolder.replace("#0", "").replace("#1", "").replace("#2", "")
            t1, t2 = experiment_name.split("-")
            cln_t1, crp_t1 = t1.split(":")
            cln_t2, crp_t2 = t2.split(":")
            cln = cln_t1 + " " + cln_t2
            crp = crp_t1 + " " + crp_t2

            diff_t1 = cln_t1 != crp_t1
            diff_t2 = cln_t2 != crp_t2
            logits_df["Corrupted"] = "NONE"
            if diff_t1:
                logits_df["Corrupted"] = "T1"
            if diff_t2:
                logits_df["Corrupted"] = "T2"
            if diff_t1 and diff_t2:
                logits_df["Corrupted"] = "BOTH"
            
            is_complex = False
            if ("top" in cln and "bottom" in cln) or ("front" in cln and "back" in cln) or ("right" in cln and "left" in cln):
                is_complex = True
            if ("top" in crp and "bottom" in crp) or ("front" in crp and "back" in crp) or ("right" in crp and "left" in crp):
                is_complex = True
            logits_df["Complex"] = is_complex

            if "residual" in logfile:
                logits_df["Component"] = "residual"
            elif "attention" in logfile:
                logits_df["Component"] = "attention"
            elif "mlp" in logfile:
                logits_df["Component"] = "mlp"

            token_mapping = {
                139: "T1",
                146: "T1",
                153: "T1",
                152: "T2",
                159: "T2",
                166: "T2",
                173: "T2",
                180: "T2",
            }
            logits_df["Transition"] = logits_df["processed_token_id"].map(token_mapping).fillna("PREEND")

            if diff_t1 and diff_t2:
                logits_df["Exp_Type"] = 2
            else:
                logits_df["Exp_Type"] = 1

            logits_df = logits_df.drop(columns=["clean_logits", "corrupt_logits", "patched_logits"])
            log_entries.append(logits_df)

df = pd.concat(log_entries, ignore_index=True)
df.loc[(df["Exp_Type"] == 1) & (df["Transition"].isin(["T1", "T2"])), "Transition"] = "T"

aggregate = (
    df.groupby(["Exp_Type", "Component", "Transition", "layer"], as_index=False)
      .agg(
          avg_normalized_logit_diff=("normalized_logit_diff", "mean"),
          avg_token_rank=("token_rank", "mean"),
      )
      .sort_values(["Exp_Type", "Component", "Transition", "layer"])
)

transition_colors = {
    "T":  "#4EA995",
    "T1": "#6BAED6",
    "T2": "#31A354",  
    "PREEND": "#DE2D26",
}
def plot_metric_by_transition(aggregate, exp_type, component, metric_col, ylim=None, invert_y=False):
    # filter
    sub = aggregate[(aggregate["Exp_Type"] == exp_type) & (aggregate["Component"] == component)].copy()
    
    sub = sub.sort_values(["Transition", "layer"])
    fig, ax = plt.subplots(figsize=(10, 5))

    for transition, g in sub.groupby("Transition", sort=False):
        g = g.sort_values("layer")
        color = transition_colors.get(transition, "black")
        ax.plot(
            g["layer"].values,
            g[metric_col].values,
            marker="o",
            label=str(transition),
            linewidth=2.5,
            markersize=3.5,
            color=color,
        )

    ax.set_xlabel("Layer", fontsize=16)
    ax.legend(fontsize=18, loc="best")
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.3)
    if metric_col == "avg_normalized_logit_diff":
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=0.8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if invert_y:
        ax.invert_yaxis()
    plt.tight_layout()
    return fig

components = ["residual", "attention", "mlp"]
exp_types = [1, 2]
for exp_type in exp_types:
    for component in components:
        fig = plot_metric_by_transition(
            aggregate=aggregate,
            exp_type=exp_type,
            component=component,
            metric_col="avg_normalized_logit_diff",
            title=f"",
            ylim=(-1.1, 1.1),
        )
        fig.savefig(f"{logfolders}/logit_diff.png", dpi=300)
        plt.close(fig)

        fig = plot_metric_by_transition(
            aggregate=aggregate,
            exp_type=exp_type,
            component=component,
            metric_col="avg_token_rank",
            title=f"",
            ylim=(15, 0.5),
        )
        fig.savefig(f"{logfolders}/rank.png", dpi=300)
        plt.close(fig)
