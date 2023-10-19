# Code organization

### A few pointers

-   [train.py](train.py) is the main script used to run experiments with LASER, Z-SGD and Signum on Cifar10 and Cifar100.
-   [sketching.py](sketching.py) and [randomk.py](randomk.py) are modified versions of train.py used to run experiments with Count-Mean Sketching and Random-K algorithms.
-   [LLM/main.py](LLM/notebook_max.py) is the script used to run the experiments on WikiText-103 with the GPT-2 model.
-   [adsgd.py](adsgd.py) is the script used to run experiments with all baselines (including A-DSGD) on MNIST.
-   [notebook_max.py](notebook_max.py) and [notebook100_max.py](notebook100_max.py) are the scripts used to generate Figure 1, 2 and 4 in the paper.
-   [notebook_max.py](notebook_power-epoch.py) was used to generate Figure 3 and 7.
-   [notebook_max.py](notebook_sv.py) was used to generate Figure 6.

### Choice of algorithm

Change `config["algorithm"]` in [train.py](train.py), [sketching.py](sketching.py) or [randomk.py](randomk.py), to experiment with the different baselines. `powersgd` corresponds to LASER, `sgd` to Z-SGD, `sketching` to Count-Mean Sketching, `randomk` to Random-K, `signum` to Signum, and `adsgd` to A-DSGD. A-DSGD requires the [Vampyre](https://github.com/GAMPTeam/vampyre) package to run; its implementation was obtained through private correspondence with the authors of the [original paper](https://arxiv.org/abs/1901.00844).
