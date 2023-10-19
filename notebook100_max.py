#%%
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

api = wandb.Api()

# 
res1 = []
for run in api.runs("w16-cifar100", {"config.algorithm": "powersgd"}):
    try:
        res1.append(
            (
                run.id,
                run.config.get("power", None),
                run.config["learning_rate"],
                run.summary["best_accuracy"],
                run.summary["_step"],
            )
        )
    except:
        pass

res2 = []
for run in api.runs("w16-cifar100", {"config.algorithm": "sgd"}):
    try:
        res2.append(
            (
                run.id,
                run.config.get("power", None),
                run.config["learning_rate"],
                run.summary["best_accuracy"],
                run.summary["_step"],
            )
        )
    except:
        pass

res3 = []
for run in api.runs("w16-cifar100", {"config.algorithm": "sketching"}):
    try:
        res3.append(
            (
                run.id,
                run.config.get("power", None),
                run.config["learning_rate"],
                run.summary["best_accuracy"],
                run.summary["_step"],
            )
        )
    except:
        pass

res4 = []
for run in api.runs("w16-cifar100", {"config.algorithm": "randomk"}):
    try:
        res4.append(
            (
                run.id,
                run.config.get("power", None),
                run.config["learning_rate"],
                run.summary["best_accuracy"],
                run.summary["_step"],
            )
        )
    except:
        pass

res5 = []
for run in api.runs("w16-cifar100", {"config.algorithm": "signum"}):
    try:
        res5.append(
            (
                run.id,
                run.config.get("power", None),
                run.config["learning_rate"],
                run.summary["best_accuracy"],
                run.summary["_step"],
            )
        )
    except:
        pass
#
df1 = pd.DataFrame(
    res1, columns=["id", "power", "learning_rate", "best_accuracy", "step"]
)
df2 = pd.DataFrame(
    res2, columns=["id", "power", "learning_rate", "best_accuracy", "step"]
)
df3 = pd.DataFrame(
    res3, columns=["id", "power", "learning_rate", "best_accuracy", "step"]
)
df4 = pd.DataFrame(
    res4, columns=["id", "power", "learning_rate", "best_accuracy", "step"]
)
df5 = pd.DataFrame(
    res5, columns=["id", "power", "learning_rate", "best_accuracy", "step"]
)

#%%
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.semilogx(df1.groupby("power")["best_accuracy"].max(), color="tab:green", marker="o", label="                          ")
ax.semilogx(df2.groupby("power")["best_accuracy"].max(), color="tab:blue", marker="x", label=" ")
ax.semilogx(df3.groupby("power")["best_accuracy"].max(), color="tab:orange", marker="^", label=" ")
ax.semilogx(df4.groupby("power")["best_accuracy"].max(), color="tab:purple", marker="s", label=" ")
#ax.semilogx(df5.groupby("power")["best_accuracy"].max(), color="tab:cyan", marker="D", label="Signum")
ax.axhline(y = 0.73, color="black", linestyle = '--', label=" ")
ax.grid(True, which="both")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
ax.legend(prop={'size': 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.savefig("cifar100-power-accuracy.pdf")