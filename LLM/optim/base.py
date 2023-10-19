from contextlib import nullcontext

import torch
import torch.nn.functional as F
import wandb
import time 
import copy
import powersgd
import numpy as np
import count_mean_sketch
from dataclasses import dataclass

from .utils import eval, get_batch, save_checkpoint


def train_base(models, opts, data
, schedulers, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path, distributed_backend, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.bfloat16)  # extra_args.dtype)
    itr, substep, best_val_loss, text_table = 0, 0, float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 

    stats = {'train_loss': [], 'val_loss': [], 'val_pp': [], 'val_acc': []}
    workers = len(models)
    num_substeps_per_epoch = len(data['train']) // (batch_size * sequence_length)
    generator = torch.Generator().manual_seed(0)
    
    if not extra_args.no_compile:
        print(f"Compiling model ...")
        for i in range(workers):
            models[i] = torch.compile(models[i]) # requires pytorch 2.0+
    
    # PowerSGD option
    if extra_args.algorithm == "powersgd":
        powersgds = []
        powersgd_config = powersgd.Config(rank=extra_args.powersgd_rank, power = extra_args.power, exponent = extra_args.exponent)
        for i in range(workers):
            powersgds.append(powersgd.PowerSGD(powersgd.params_in_optimizer(opts[i]), powersgd_config))
            if i > 0:
                powersgds[i].set_pq(*powersgds[0].get_pq())
    elif extra_args.algorithm == "sketching":
        sketching = count_mean_sketch.CountSketchAggregator(extra_args.compression_factor, "one by one")

    for model in models:
        model.train()

    t0 = time.time()
    while itr < iterations:
        for i in range(workers):
            for microstep_idx in range(acc_steps):  # gradient accumulation
                x, y = get_batch(data['train'], sequence_length, batch_size, device=extra_args.device)
                with type_ctx:
                    with distributed_backend.get_context_for_microstep_forward(model=models[i], microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                        outputs = models[i](x, targets=y)
                
                loss = outputs['loss']
                loss.backward()
                substep += 1
        
        if extra_args.algorithm == "sgd":
            params = []
            for i in range(workers):
                params.append(powersgd.params_in_optimizer(opts[i]))
            device = params[0][0].device
            # Power allocation
            power_per_param = allocate_power(params, extra_args.power)
            noise_coeffs = noise_coefficients(params)
            # Sum gradients up
            for i, p in enumerate(params[0]):
                g = torch.zeros_like(p.grad, device=device)
                for w in range(workers):
                    g.add_(params[w][i].grad.data, alpha=1/workers)
                p.grad = g
            
            # Apply channel corruption
            if extra_args.power > 0:
                apply_channel(params, power_per_param, noise_coeffs)
            
            # Update model parameters
            for i in range(1,workers):
                for j in range(len(params[0])):
                    params[i][j].grad = params[0][j].grad
            for opt in opts:
                opt.step()
            for opt in opts:
                opt.zero_grad(set_to_none=True)
        elif extra_args.algorithm == "powersgd":
            optimizer_step(powersgds, opts, extra_args.power)
        elif extra_args.algorithm == "sketching":
            params = []
            for i in range(workers):
                params.append(powersgd.params_in_optimizer(opts[i]))
            device = params[0][0].device
            grads = []
            for i in range(workers):
                grads.append([p.grad.data for p in params[i]])
            avg_grads = sketching.aggregate(grads, extra_args.power)
            # Set average grads
            for param in params:
                for p, g in zip(param, avg_grads):
                    p.grad = g
            # Update model parameters
            for opt in opts:
                opt.step()
            for opt in opts:
                opt.zero_grad(set_to_none=True)
        elif extra_args.algorithm == "randomk":
            params = []
            for i in range(workers):
                params.append(powersgd.params_in_optimizer(opts[i]))
            device = params[0][0].device
            grads = []
            for i in range(workers):
                grads.append([p.grad.data for p in params[i]])
            avg_grads = randomk_aggregate(grads, generator, extra_args.power, extra_args.compression_factor)
            # Set average grads
            for param in params:
                for p, g in zip(param, avg_grads):
                    p.grad = g
            # Update model parameters
            for opt in opts:
                opt.step()
            # Put back error feedback
            for i in range(workers):
                for (p, g) in zip(params[i], grads[i]):
                    p.grad = g
        elif extra_args.algorithm == "signum":
            params = []
            for i in range(workers):
                params.append(powersgd.params_in_optimizer(opts[i]))
            device = params[0][0].device
            grads = []
            for i in range(workers):
                grads.append([p.grad.data for p in params[i]])
            avg_grads = signum_aggregate(grads, extra_args.power)
            # Set average grads
            for param in params:
                for p, g in zip(param, avg_grads):
                    p.grad = g
            # Update model parameters
            for opt in opts:
                opt.step()
            for opt in opts:
                opt.zero_grad(set_to_none=True)
        else:
            raise ValueError("Unknown algorithm")
        
        for scheduler in schedulers:
            scheduler.step()
        itr += 1

        model = models[0]
        opt = opts[0]
        scheduler = schedulers[0]
        if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
            if distributed_backend.is_master_process():
                t1 = time.time()
                dt = t1 - t0
                epoch = substep//num_substeps_per_epoch

                model.eval()
                train_loss = loss.detach().cpu().item()
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
                val_acc, val_loss, val_perplexity = eval(model, data['val'], sequence_length, batch_size,
                                                         extra_args.device, max_num_batches=24, ctx=type_ctx)

                print_string = f"{epoch}/{itr} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
                print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
                if scheduler is not None:
                    print_string += f" [lr] {current_lr:.5f}"
                print(print_string)

                if extra_args.wandb:
                    wandb.log({
                        "iter": itr,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/perplexity": val_perplexity,
                        "val/acc": val_acc,
                        "lr": current_lr,
                    })

                    if extra_args.eval_seq_prefix != 'none' and (itr % (eval_freq * 5) == 0 or itr == iterations):
                        if text_table is None:
                            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

                        out_str = distributed_backend.get_raw_model(model).generate_from_string(
                            extra_args.eval_seq_prefix, max_new_tokens=40, temperature=0.9, top_k=None)
                        text_table.add_data(itr, val_perplexity, out_str)
                        # why a copy? see github.com/wandb/wandb/issues/2981
                        wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})

                model.train()
                t0 = time.time()

    if distributed_backend.is_master_process():
        print(f"saving checkpoint to {ckpt_path}")
        save_checkpoint(distributed_backend=distributed_backend,
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
                        itr=itr,
                        ckpt_path=ckpt_path)

    return stats

@dataclass
class RandomkCompressed:
    seed: int
    original_shape: torch.Size
    vector: torch.Tensor

def randomk_aggregate(gradients, generator, power, cf):
    device = gradients[0][0].device
    workers = len(gradients)
    out = [torch.zeros_like(g) for g in gradients[0]]
    mean_grads = []
    for g in gradients[0]:
        compressed_size = int(cf * g.numel())
        if compressed_size == 0:
            compressed_size = 1
        mean_grads.append(torch.zeros(compressed_size).to(device))
    shared = [int(torch.randint(1000000000, (1,), generator=generator)) for g in gradients[0]]
    power_per_param = torch.zeros(len(gradients[0])).to(device)
    noise_coeffs = torch.zeros(len(gradients[0])).to(device)

    for i in range(workers):
        worker_power = torch.zeros(len(gradients[0])).to(device)
        for j, g in enumerate(gradients[i]):
            compressed_size = int(cf * g.numel())
            if compressed_size == 0:
                compressed_size = 1
            compressed = compress(g, compressed_size, shared[j])
            worker_power[j] = torch.linalg.norm(compressed.vector)
            # Average gradient
            mean_grads[j].add_(compressed.vector, alpha=1/workers)
            # Error feedback
            g.add_(uncompress(compressed), alpha=-1)
        noise_coeffs.copy_(torch.maximum(noise_coeffs, worker_power))
        power_per_param.add_(F.normalize(worker_power, p = 1, dim = 0), alpha=power/workers)

    # Add noise to compressed grads
    for g, p, c in zip(mean_grads, power_per_param, noise_coeffs):
        g.add_(torch.randn_like(g), alpha=c/(workers*torch.sqrt(p)))
    
    # Reconstruct grads
    for i, g in enumerate(mean_grads):
        out[i].add_(uncompress(RandomkCompressed(seed=shared[i], original_shape=out[i].shape, vector=g)))

    return out


def compress(tensor: torch.Tensor, compressed_size: int, seed: int) -> RandomkCompressed:
    device = tensor.device
    flat_tensor = tensor.view(-1)
    generator = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randperm(tensor.numel(), generator=generator, device=device)[:compressed_size]
    compressed = torch.zeros(compressed_size, device=device)
    compressed.add_(flat_tensor[indices])

    return RandomkCompressed(seed=seed, original_shape=tensor.shape, vector=compressed)


def uncompress(compressed: RandomkCompressed) -> torch.Tensor:
    device = compressed.vector.device
    uncompressed = torch.zeros(compressed.original_shape.numel(), device=device)
    generator = torch.Generator(device=device).manual_seed(compressed.seed)
    indices = torch.randperm(compressed.original_shape.numel(), generator=generator, device=device)[:compressed.vector.numel()]
    uncompressed[indices] = compressed.vector

    return uncompressed.view(*compressed.original_shape)


def signum_aggregate(gradients, power):
    device = gradients[0][0].device
    workers = len(gradients)
    avg_grads = [torch.zeros_like(g) for g in gradients[0]]
    out_grads = [torch.zeros_like(g) for g in gradients[0]]
    power_per_param = F.normalize(torch.tensor([np.sqrt(g.numel()) for g in gradients[0]], device=device), p = 1, dim = 0) * power
    noise_coeffs = [np.sqrt(g.numel()) for g in gradients[0]]

    # Sum up the gradients
    for w in gradients:
        for g, avg in zip(w, avg_grads):
            avg.add_(torch.sign(g), alpha=1/workers)

    # Add noise to compressed grads
    for avg, p, c in zip(avg_grads, power_per_param, noise_coeffs):
        avg.add_(torch.randn_like(avg), alpha=c/(workers*torch.sqrt(p)))
    
    # Reconstruct grads
    for o, avg in zip(out_grads, avg_grads):
        o.add_(torch.sign(avg))

    return out_grads


def allocate_power(params, power):
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
    grads = [p.grad for p in params[0]]
    for g, c, p in zip(grads, noise_coeffs, power_per_param):
        # Apply noise in place
        if p > 0:
            g.add_(
                torch.randn_like(g),
                alpha=c / (len(params) * torch.sqrt(p))
            )

def optimizer_step(powersgds, optimizers, power):
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
            avg.add_(p, alpha=power/workers)
        avg_uncompressed_power.add_(uncompressed_power, alpha=power/workers)
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
    for i in range(workers):
        optimizers[i].step()
    #for w in params:
    #    for i in range(len(w)):
    #        w[i].data = params[0][i].data
    
    # Put back the error buffer as the parameter's gradient
    for i in range(workers):
        for (p, g) in zip(params[i], grads[i]):
            p.grad = g