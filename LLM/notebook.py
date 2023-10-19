#%%
import wandb
import pandas as pd
from matplotlib import pyplot as plt
api = wandb.Api()


# %% Effect of power on accuracy (300 epochs, Cifar-10, ResNet-18)
res = []
for run in api.runs("my-project",{"$and": [{"group": "power-lr-test-30000"}, {"config.algorithm": "powersgd"}, {"config.powersgd_rank": 4}]}):
    try:
        res.append((run.id, run.config.get("power", None), run.config["lr"], run.summary["val/perplexity"]))
    except:
        pass

df = pd.DataFrame(res, columns=["id", "power", "lr", "val/perplexity"])
piv = df.pivot_table(index="power", columns="lr", values="val/perplexity")
fig, ax = plt.subplots(figsize=(10, 10))
v = ax.matshow(piv, vmin=1, vmax=1000)
ax.set_xticks(range(len(piv.columns)), piv.columns)
ax.set_yticks(range(len(piv.index)), piv.index)
ax.xaxis.set_ticks_position("bottom")
ax.set(xlabel="Learning rate", ylabel="Power")
ax.set_title("Effect of power on perplexity (PowerSGD, Rank 4, NanoGPT, 30k iterations)")
fig.colorbar(v, ax=ax, label="Final perplexity")
for i in range(len(piv.index)):
    for j in range(len(piv.columns)):
        ax.text(j, i, f"{piv.iloc[i, j]:.1f}", ha="center", va="center", color="w")
fig.savefig("gpt-powersgd-30k-rank4.png", dpi=300)

'''# %% PowerSGD varying rank (very high power)
res = []
for run in api.runs("federated-codes", {"group": "psgdstudy"}):
    try:
        res.append((run.id, run.config.get("powersgd_rank", None), run.config["learning_rate"], run.summary["best_accuracy"], run.summary["_step"]))
    except:
        pass

df = pd.DataFrame(res, columns=["id", "powersgd_rank", "learning_rate", "best_accuracy", "step"])
piv = df.pivot_table(index="powersgd_rank", columns="learning_rate", values="best_accuracy")
fig, ax = plt.subplots(figsize=(8, 6))
v = ax.matshow(piv, vmin=0.6, vmax=0.85)
ax.set_xticks(range(len(piv.columns)), piv.columns)
ax.set_yticks(range(len(piv.index)), piv.index.astype(int))
ax.xaxis.set_ticks_position("bottom")
ax.set(xlabel="Learning rate", ylabel="PowerSGD Rank")
ax.set_title("Effect of PowerSGD rank (100 epochs, Cifar-10, ResNet-18)")
fig.colorbar(v, ax=ax, label="Final accuracy (exponential moving average)")
for i in range(len(piv.index)):
    for j in range(len(piv.columns)):
        ax.text(j, i, f"{100*piv.iloc[i, j]:.0f}%", ha="center", va="center", color="w")
fig.savefig("powersgd_rankstudy.png", dpi=300)
# %%'''
