#!/usr/bin/env python3

import json
import os
from copy import deepcopy
from typing import Any, Iterable

import models
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends
import torch.utils.data
import torchvision
import utils.accumulators
import wandb
import powersgd

CHECKPOINT_FILENAME = "checkpoint.pt"

config: dict[str, Any] = dict(
    dataset="Cifar100",
    model="resnet18",
    optimizer="SGD",
    decay_at_epochs=[150, 250],
    decay_with_factor=10.0,
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    batch_size=128,
    power=1000,
    num_epochs=200,
    seed=42,
    ema_alpha=0.99,
    algorithm="signum",
    powersgd_rank=4,
    power_allocation="norm",
    exponent=1,
    workers=16
)

# Override config values from environment variables:
for key, default_value in config.items():
    if os.getenv(key) is not None:
        if not isinstance(default_value, str):
            try:
                config[key] = json.loads(os.getenv(key, ""))
            except json.decoder.JSONDecodeError:
                print(f"Failed to decode {key} from environment variable: {os.getenv(key)}.")
                exit(1)
        else:
            config[key] = os.getenv(key, "")


def main():
    wandb.init(config=config, resume="allow")

    # Set the seed
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # We will run on CUDA if there is a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure the dataset, model and the optimizer based on the global
    # `config` dictionary.
    workers = config["workers"]
    training_loaders, test_loader = get_dataset(workers=workers)
    models = []
    optimizers = []
    schedulers = []
    criterions = []
    models.append(get_model(device))
    for i in range(1,workers):
        models.append(deepcopy(models[0]))
    for i in range(workers):
        opt, sch = get_optimizer(models[i].parameters())
        optimizers.append(opt)
        schedulers.append(sch)
        criterions.append(torch.nn.CrossEntropyLoss())
    params = [list(model.parameters()) for model in models]

    # We keep track of the best accuracy so far to store checkpoints
    best_accuracy_so_far = utils.accumulators.Max()

    mean_test_accuracy = utils.accumulators.Mean()

    start_epoch = 0

    if wandb.run is not None and wandb.run.resumed:
        wandb.restore("checkpoint.pt")
        checkpoint = torch.load("checkpoint.pt")
        for i in range(workers):
            models[i].load_state_dict(checkpoint["model_state_dict"])
            schedulers[i].load_state_dict(checkpoint["scheduler_state_dict"])
            optimizers[i].load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_accuracy_so_far.add(checkpoint["best_accuracy_so_far"])
        print(f"Resuming from epoch {start_epoch}.")

    average_weights = utils.accumulators.EMA(alpha=config["ema_alpha"])
    if config["algorithm"] == "powersgd":
        powersgd_config = powersgd.Config(rank=config["powersgd_rank"], power = config["power"], exponent = config["exponent"])
        powersgds = []
        for i in range(workers):
            powersgds.append(powersgd.PowerSGD(list(models[i].parameters()), powersgd_config))
            if i > 0:
                powersgds[i].set_pq(*powersgds[0].get_pq())
 
    for epoch in range(start_epoch, config["num_epochs"]):
        # Enable training mode (automatic differentiation + batch norm)
        for model in models:
            model.train()
        
        # Keep track of statistics during training
        mean_train_accuracy = utils.accumulators.Mean()
        mean_train_loss = utils.accumulators.Mean()

        assert len(training_loaders[0]) == len(training_loaders[workers-1])
        for batch_set in zip(*training_loaders):
            for i in range(workers):
                batch_x, batch_y = batch_set[i][0].to(device), batch_set[i][1].to(device)
                # Compute gradients for the worker
                if config["algorithm"] == "sgd" or config["algorithm"] == "signum":
                    optimizers[i].zero_grad()
                prediction = models[i](batch_x)
                loss = criterions[i](prediction, batch_y)
                acc = accuracy(prediction, batch_y)
                loss.backward()
            
            # Do an optimizer step
            if config["algorithm"] == "sgd":
                # Power allocation
                power_per_param = allocate_power(params, config["power"], config["power_allocation"])
                noise_coeffs = noise_coefficients(params)
                # Sum gradients up
                for i, p in enumerate(params[0]):
                    g = torch.zeros_like(p.grad, device=device)
                    for w in range(workers):
                        g.add_(params[w][i].grad.data, alpha=1/workers)
                    p.grad = g
                
                # Apply channel corruption
                if config["power"] > 0:
                    apply_channel(params, power_per_param, noise_coeffs)
                
                # Update model parameters
                optimizers[0].step()
                for w in params:
                    for i in range(len(w)):
                        w[i].data = params[0][i].data
            elif config["algorithm"] == "powersgd":
                optimizer_step(powersgds, optimizers)
            elif config["algorithm"] == "signum":
                grads = []
                for i in range(workers):
                    grads.append([p.grad.data for p in params[i]])
                avg_grads = signum_aggregate(grads, config["power"])
                # Set average grads
                for param in params:
                    for p, g in zip(param, avg_grads):
                        p.grad = g
                for opt in optimizers:
                    opt.step()
            else:
                raise ValueError("Unknown algorithm")

            average_weights.add(models[0].state_dict())

            # Store the statistics
            mean_train_loss.add(loss.item(), weight=len(batch_x))
            mean_train_accuracy.add(acc.item(), weight=len(batch_x))

        # Update the optimizers' learning rates
        for i in range(workers):
            schedulers[i].step()

        # Log training stats
        wandb.log(
            {
                "train/accuracy": mean_train_accuracy.value(),
                "train/loss": mean_train_loss.value(),
            },
            step=epoch + 1,
        )

        # Evaluation
        models[0].eval()
        mean_test_accuracy = utils.accumulators.Mean()
        mean_test_loss = utils.accumulators.Mean()
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = models[0](batch_x)
            loss = criterions[0](prediction, batch_y)
            acc = accuracy(prediction, batch_y)
            mean_test_loss.add(loss.item(), weight=len(batch_x))
            mean_test_accuracy.add(acc.item(), weight=len(batch_x))

        # Log test stats
        wandb.log(
            {
                "test/accuracy": mean_test_accuracy.value(),
                "test/loss": mean_test_loss.value(),
            },
            step=epoch + 1,
        )

        # EMA model evaluation
        ema_model = deepcopy(models[0])
        ema_model.load_state_dict(average_weights.value())
        ema_model.eval()
        mean_test_accuracy = utils.accumulators.Mean()
        mean_test_loss = utils.accumulators.Mean()
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = ema_model(batch_x)
            loss = criterions[0](prediction, batch_y)
            acc = accuracy(prediction, batch_y)
            mean_test_loss.add(loss.item(), weight=len(batch_x))
            mean_test_accuracy.add(acc.item(), weight=len(batch_x))

        # Log test stats
        wandb.log(
            {
                "test/ema_accuracy": mean_test_accuracy.value(),
                "test/ema_loss": mean_test_loss.value(),
            },
            step=epoch + 1,
        )

        best_accuracy_so_far.add(mean_test_accuracy.value())
        wandb.summary["best_accuracy"] = best_accuracy_so_far.value()

        torch.save(
            {
                "epoch": epoch + 1,
                "optimizer_state_dict": optimizers[0].state_dict(),
                "scheduler_state_dict": schedulers[0].state_dict(),
                "model_state_dict": models[0].state_dict(),
                "best_accuracy_so_far": best_accuracy_so_far.value(),
            },
            CHECKPOINT_FILENAME,
        )
        wandb.save(CHECKPOINT_FILENAME)

def signum_aggregate(gradients, power):
    device = gradients[0][0].device
    workers = len(gradients)
    avg_grads = [torch.zeros_like(g) for g in gradients[0]]
    out_grads = [torch.zeros_like(g) for g in gradients[0]]
    # Power allocation
    power_per_param = torch.zeros(len(gradients[0]), device=device)
    noise_coeffs = torch.zeros(len(gradients[0]), device=device)
    for w in gradients:
        norms = torch.tensor([torch.linalg.norm(g) for g in w], device=device)
        power_per_param.add_(F.normalize(norms, p = 1, dim = 0) * power, alpha=1/workers)
        noise_coeffs = torch.maximum(noise_coeffs, norms)
    # Sum up the gradients
    for w in gradients:
        for g, avg in zip(w, avg_grads):
            avg.add_(torch.sign(g), alpha=1/workers)
    # Add noise to compressed grads
    for avg, p, c in zip(avg_grads, power_per_param, noise_coeffs):
        if p > 0:
            avg.add_(torch.randn_like(avg), alpha=c/(workers*torch.sqrt(p)))
    # Reconstruct grads
    for o, avg in zip(out_grads, avg_grads):
        o.add_(torch.sign(avg))

    return out_grads

def allocate_power(params, power, power_allocation):
    """Implemented norm allocation only"""
    device = params[0][0].device
    power_per_grad = torch.zeros(len(params[0]), device=device)
    for worker_params in params:
        power_per_grad.add_(F.normalize(torch.Tensor([torch.linalg.norm(p.grad) for p in worker_params]).to(device), p = 1, dim = 0), alpha=power/len(params))

    return power_per_grad

def noise_coefficients(params):
    noise_coeffs = []
    for i in range(len(params[0])):
        noise_coeffs.append(max([torch.linalg.norm(w[i].grad) for w in params]))
    
    return noise_coeffs

def apply_channel(params, power_per_param, noise_coeffs):
    """Apply noise to gradients, in place."""
    grads = [p.grad for p in params[0]]
    for g, c, p in zip(grads, noise_coeffs, power_per_param):
        # Apply noise in place
        if p > 0:
            g.add_(
                torch.randn_like(g),
                alpha=c / (len(params) * torch.sqrt(p))
            )

def accuracy(predicted_logits, reference):
    """Compute the ratio of correctly predicted labels"""
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


def get_dataset(
    test_batch_size=1000,
    shuffle_train=True,
    data_root=os.getenv("DATA_DIR", "./data"),
    workers=1
):
    """
    Create dataset loaders for the chosen dataset
    :return: Tuple (training_loader, test_loader)
    """
    dataset = config["dataset"]
    if dataset == "Cifar10":
        dataset = torchvision.datasets.CIFAR10
    elif dataset == "Cifar100":
        dataset = torchvision.datasets.CIFAR100
    else:
        raise ValueError("Unexpected value for config[dataset] {}".format(dataset))

    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)

    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    training_set = dataset(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_set = dataset(
        root=data_root, train=False, download=True, transform=transform_test
    )

    training_set = torch.utils.data.Subset(training_set, torch.randperm(len(training_set)))
    len_per_worker = int(len(training_set) / workers)
    training_sets = []
    for i in range(workers):
        training_sets.append(torch.utils.data.Subset(training_set, range(i*len_per_worker, (i+1)*len_per_worker)))

    training_loaders = []
    for i in range(workers):
        training_loaders.append(torch.utils.data.DataLoader(
        training_sets[i],
        batch_size=config["batch_size"],
        shuffle=shuffle_train,
    ))
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False
    )

    return training_loaders, test_loader

def get_optimizer(model_parameters):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: Tuple (optimizer, scheduler)
    """
    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError("Unexpected value for optimizer")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["decay_at_epochs"],
        gamma=1.0 / config["decay_with_factor"],
    )

    return optimizer, scheduler


def get_model(device):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """
    num_classes = 100 if config["dataset"] == "Cifar100" else 10

    model = {
        "vgg11": lambda: models.vgg11(),
        "vgg13": lambda: models.vgg13(),
        "vgg16": lambda: models.vgg16(),
        "vgg19": lambda: models.vgg19(),
        "efficientnet_b0": lambda: torchvision.models.efficientnet_b0(
            num_classes=num_classes
        ),
        "resnet18": lambda: models.ResNet18(num_classes=num_classes),
        "resnet34": lambda: models.ResNet34(num_classes=num_classes),
        "resnet50": lambda: models.ResNet50(num_classes=num_classes),
        "resnet101": lambda: models.ResNet101(num_classes=num_classes),
        "resnet152": lambda: models.ResNet152(num_classes=num_classes),
    }[config["model"]]()

    model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)

    return model

def optimizer_step(powersgds, optimizers):
    """
    Aggregate gradients across workers using `aggregator`,
    and then take an optimizer step using the aggregated gradient.
    """
    params = []
    grads = []
    workers = len(powersgds)
    for i in range(workers):
        params.append(powersgd.params_in_optimizer(optimizers[i]))
        grads.append([p.grad.data for p in params[i]])
    device = params[0][0].device
    # Update p's and q's and average
    avg_ps = [torch.zeros_like(p, device=device) for p in powersgds[0].get_pq()[0]]
    avg_qs = [torch.zeros_like(q, device=device) for q in powersgds[0].get_pq()[1]]
    avg_uncompressed_grads = [torch.zeros_like(g, device=device) for g in powersgds[0].get_uncompressed_grads(grads[0])]
    avg_compressed_power = [torch.zeros(p_batch.shape[0], device=device) for p_batch in powersgds[0].get_pq()[0]]
    avg_uncompressed_power = torch.zeros(len(avg_uncompressed_grads), device=device)
    noise_coeff_compressed = [torch.zeros(p_batch.shape[0], device=device) for p_batch in powersgds[0].get_pq()[0]]
    noise_coeff_uncompressed = torch.zeros(len(avg_uncompressed_grads), device=device)

    for i in range(workers):
        compressed_grads, uncompressed_grads = powersgds[i]._split(grads[i])
        ps, qs, compressed_power, uncompressed_power = powersgds[i].update(compressed_grads, uncompressed_grads)
        for j in range(len(avg_ps)):
            avg_ps[j].add_(ps[j], alpha=1/workers)
            avg_qs[j].add_(qs[j], alpha=1/workers)
        for j, g in enumerate(uncompressed_grads):
            avg_uncompressed_grads[j].add_(g, alpha=1/workers)
        for avg, p in zip(avg_compressed_power, compressed_power):
            avg.add_(p, alpha=config["power"]/workers)
        avg_uncompressed_power.add_(uncompressed_power, alpha=config["power"]/workers)
        if powersgds[0].get_counter() % 2 == 0:
            for c, q in zip(noise_coeff_compressed, qs):
                c.copy_(torch.maximum(c, torch.linalg.norm(q, dim=(1,2))))
        else:
            for c, p in zip(noise_coeff_compressed, ps):
                c.copy_(torch.maximum(c, torch.linalg.norm(p, dim=(1,2))))
        noise_coeff_uncompressed = torch.maximum(noise_coeff_uncompressed, torch.Tensor([torch.linalg.norm(g) for g in uncompressed_grads]).to(device))
        # Zero uncompressed gradients and merge
        for g in uncompressed_grads:
            g.zero_()
        grads[i] = powersgds[i]._merge(compressed_grads, uncompressed_grads)
    # Apply channel noise
    if powersgds[0].get_counter() % 2 == 0:
        for g_batch, p_batch, c_batch in zip(avg_qs, avg_compressed_power, noise_coeff_compressed):
            for g, p, c in zip(g_batch,p_batch,c_batch):
                g.add_(torch.randn_like(g), alpha=c/(workers*torch.sqrt(p)))
    else:
        for g_batch, p_batch, c_batch in zip(avg_ps, avg_compressed_power, noise_coeff_compressed):
            for g, p, c in zip(g_batch,p_batch,c_batch):
                g.add_(torch.randn_like(g), alpha=c/(workers*torch.sqrt(p)))
    for g, p, c in zip(avg_uncompressed_grads, avg_uncompressed_power, noise_coeff_uncompressed):
        g.add_(
            torch.randn_like(g),
            alpha= c/ (workers * torch.sqrt(p))
        )
    
    # Temporarily set parameter's gradients to the aggregated values and update parameters
    for i in range(workers):
        powersgds[i].set_grads(params[i], avg_ps, avg_qs, avg_uncompressed_grads)
    optimizers[0].step()
    for w in params:
        for i in range(len(w)):
            w[i].data = params[0][i].data
    
    # Put back the error buffer as the parameter's gradient
    for i in range(workers):
        for (p, g) in zip(params[i], grads[i]):
            p.grad = g

if __name__ == "__main__":
    main()
