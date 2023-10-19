#%%
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

api = wandb.Api()

# 
res1 = []
for run in api.runs("power-epoch", {"config.algorithm": "powersgd", "config.power_per_epoch": "constant", "config.backwards": 1}):
    try:
        res1.append(
            (
                run.id,
                run.config.get("power", None),
                run.summary["best_accuracy"],
                run.summary["_step"],
            )
        )
    except:
        pass

res2 = []
for run in api.runs("power-epoch", {"config.algorithm": "powersgd", "config.power_per_epoch": "linear", "config.backwards": 0}):
    try:
        res2.append(
            (
                run.id,
                run.config.get("power", None),
                run.summary["best_accuracy"],
                run.summary["_step"],
            )
        )
    except:
        pass

res3 = []
for run in api.runs("power-epoch", {"config.algorithm": "powersgd", "config.power_per_epoch": "step", "config.backwards": 0}):
    try:
        res3.append(
            (
                run.id,
                run.config.get("power", None),
                run.summary["best_accuracy"],
                run.summary["_step"],
            )
        )
    except:
        pass

res4 = []
for run in api.runs("power-epoch", {"config.algorithm": "sgd", "config.power_per_epoch": "constant", "config.backwards": 1}):
    try:
        res4.append(
            (
                run.id,
                run.config.get("power", None),
                run.summary["best_accuracy"],
                run.summary["_step"],
            )
        )
    except:
        pass

res5 = []
for run in api.runs("power-epoch", {"config.algorithm": "sgd", "config.power_per_epoch": "linear", "config.backwards": 0}):
    try:
        res5.append(
            (
                run.id,
                run.config.get("power", None),
                run.summary["best_accuracy"],
                run.summary["_step"],
            )
        )
    except:
        pass

res6 = []
for run in api.runs("power-epoch", {"config.algorithm": "sgd", "config.power_per_epoch": "step", "config.backwards": 0}):
    try:
        res6.append(
            (
                run.id,
                run.config.get("power", None),
                run.summary["best_accuracy"],
                run.summary["_step"],
            )
        )
    except:
        pass

#
df1 = pd.DataFrame(
    res1, columns=["id", "power", "best_accuracy", "step"]
)
df2 = pd.DataFrame(
    res2, columns=["id", "power", "best_accuracy", "step"]
)
df3 = pd.DataFrame(
    res3, columns=["id", "power", "best_accuracy", "step"]
)
df4 = pd.DataFrame(
    res4, columns=["id", "power", "best_accuracy", "step"]
)
df5 = pd.DataFrame(
    res5, columns=["id", "power", "best_accuracy", "step"]
)
df6 = pd.DataFrame(
    res6, columns=["id", "power", "best_accuracy", "step"]
)

#%%
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.semilogx(df1.groupby("power")["best_accuracy"].max(), color="tab:green", marker="o", label="                        ") #LASER, constant
ax.semilogx(df2.groupby("power")["best_accuracy"].max(), color="tab:blue", marker="x", label="           ") #LASER, linear
ax.semilogx(df3.groupby("power")["best_accuracy"].max(), color="tab:orange", marker="^", label="   ") #LASER, step
ax.semilogx(df4.groupby("power")["best_accuracy"].max(), color="tab:purple", marker="s", label="   ") #Z-SGD, constant
ax.semilogx(df5.groupby("power")["best_accuracy"].max(), color="tab:cyan", marker="D", label="   ") #Z-SGD, linear
ax.semilogx(df6.groupby("power")["best_accuracy"].max(), color="tab:brown", marker="h", label="   ") #Z-SGD, step
ax.grid(True, which="both")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
ax.legend(prop={'size': 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#ax.set(xlabel="Power", ylabel="Final accuracy after 150 epochs")
fig.savefig("power-epoch-full.pdf")