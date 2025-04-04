import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from tqdm import tqdm
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
import time

########################################################################################
########################################################################################

def get_loss_and_accuracy(logits, targets, eq_positions, mask, reduction='mean'):
    """
    Computes the mean negative log-likelihood loss and the accuracy on the right-hand side (RHS)
    of each equation in the mini-batch.

    The equation can be : 
        - "[BOS] [a] [+] [b] [=] [r] [EOS] [PAD] [PAD]", in that case target is "[a] [+] [b] [=] [r] [EOS] [PAD] [PAD]"
        - "[BOS] [a] [+] [b] [+] [c] [=] [r] [EOS]", in that case target is "[a] [+] [b] [+] [c] [=] [r] [EOS]"

    Let :
        - B : batch size
        - S : sequence length
        - V : vocabulary size
    
    Parameters
    ----------
    logits : torch.FloatTensor of shape (B, S, V)
        A tensor containing the logits of the next token for all positions in each sequence of the mini-batch.
    targets : torch.LongTensor of shape (B, S)
        A tensor containing the target next tokens for all positions in each sequence of the mini-batch.
    eq_positions : torch.LongTensor of shape (B,)
        The position of the '=' token in each sequence (each sample has exactly one '=').
    mask : torch.LongTensor of shape (B, S)
        A mask indicating valid tokens (1 if valid, 0 for PAD tokens).
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        - 'none': no reduction will be applied
        - 'mean': average the output of the batch dimension. 
        - 'sum': sum the output of the batch dimension.
        
    Returns
    -------
    loss : torch.Tensor of shape (1,) or (B,) depending on the reduction
        The negative log-likelihood loss computed over the valid (non-PAD) RHS tokens.
    accuracy : torch.Tensor of shape (1,) or (B,) depending on the reduction
        The accuracy over the batch where a sequence is counted as correct only if 
        all valid RHS tokens are predicted correctly.
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device
    
    eq_positions = eq_positions.to(device)
    mask = mask.to(device)
    targets = targets.to(device)
    
    position_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    position_indices = position_indices.expand(batch_size, -1)
    eq_positions = eq_positions.unsqueeze(-1)
    RHS_positions_mask = position_indices > eq_positions
    valid_RHS_positions_mask = RHS_positions_mask & (mask == 1)
    
    RHS_logits = logits[valid_RHS_positions_mask]
    RHS_targets = targets[valid_RHS_positions_mask]
    
    log_probs = F.log_softmax(RHS_logits, dim=-1)
    RHS_targets_unsqueezed = RHS_targets.unsqueeze(-1)
    token_losses = -log_probs.gather(1, RHS_targets_unsqueezed)
    token_losses = token_losses.squeeze(1)
    
    sequence_losses = torch.zeros(batch_size, device=device)
    sequence_accuracies = torch.ones(batch_size, device=device)
    
    n_valid_RHS_positions = valid_RHS_positions_mask.sum(dim=1)
    valid_sequences_mask = n_valid_RHS_positions > 0

    n_valid_RHS_positions = n_valid_RHS_positions[valid_sequences_mask].tolist()
    split_losses = torch.split(token_losses, n_valid_RHS_positions)
    split_preds = torch.split(RHS_logits.argmax(dim=-1), n_valid_RHS_positions)
    split_targets = torch.split(RHS_targets, n_valid_RHS_positions)
    
    mean_losses = torch.stack([x.mean() for x in split_losses])
    sequence_losses[valid_sequences_mask] = mean_losses

    comparisons = [p == t for p, t in zip(split_preds, split_targets)]
    all_correct_sequences = [(comparison.all()).float() for comparison in comparisons]
    stacked_accuracies = torch.stack(all_correct_sequences)
    sequence_accuracies[valid_sequences_mask] = stacked_accuracies
    
    if reduction == 'mean':
        mean_loss = sequence_losses[valid_sequences_mask].mean()
        mean_accuracy = sequence_accuracies[valid_sequences_mask].mean()
        return (mean_loss.cpu(), mean_accuracy.cpu())
    
    elif reduction == 'sum':
        sum_loss = sequence_losses.sum()
        sum_accuracy = sequence_accuracies.sum()
        return sum_loss.cpu(), sum_accuracy.cpu()
    return sequence_losses.cpu(), sequence_accuracies.cpu()


########################################################################################
########################################################################################
  
@torch.no_grad()
def eval_model(model, loader, device) :
    model.eval()
    acc = 0
    loss = 0
    n = 0
    for batch in loader:
        batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits, *_ = model(batch_x) # (B, S, V)
        batch_loss, batch_acc = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)
        n += batch_x.shape[0]
        loss += batch_loss.item() * batch_x.shape[0]
        acc += batch_acc * batch_x.shape[0]

    # Calculate L2 norm of model parameters θ^(t)
    params_vector = nn_utils.parameters_to_vector(model.parameters())
    l2_norm = params_vector.norm(2).item()

    ##########
    # You can add more metrics in the dictionary (e.g., l2 norm of the parameters, etc.) 
    ##########

    return {"loss" : loss / n, "accuracy": acc / n, "l2_norm": l2_norm}
    
########################################################################################
########################################################################################


def train(
    model, train_loader, train_loader_for_eval, test_loader, optimizer, scheduler, device, 
    exp_name:str, checkpoint_path:str,
    n_steps:int, eval_first:int=0, eval_period:int=1, print_step:int=1, save_model_step:int=1,  save_statistic_step:int=1,  
    verbose=True,
    ):
    """
    model (nn.Module) : The model to train
    train_loader (DataLoader) : Training data loader
    train_loader_for_eval (DataLoader) : Training data loader (for evaluation)
    test_loader (DataLoader) : Test/Val data loader
    optimizer (Optimizer) : Optimizer
    device (str) : Device (cpu, cuda, cuda:0, etc)
    exp_name (str) : experiment name
    checkpoint_path (str) : Path to save the model checkpoints ("/path/to/experiment")
    n_steps (int) : Number of training steps
    eval_first (int) : Number of consecutive evaluation step at the beginning of training
    eval_period (int) : Evaluation frequency
    print_step (int) : Print frequency
    save_model_step (int) : Step interval to save model checkpoints
    save_statistic_step (int) : Step interval to save statistics (train/test loss, accuracy, etc.)
    verbose (bool) : Verbosity of the training
    """

    ##############
    # Checkpoint path
    os.makedirs(checkpoint_path, exist_ok=True)

    ##############
    # Number of training epochs
    total_epochs = (n_steps + len(train_loader) - 1) // len(train_loader)
    n_steps = total_epochs * len(train_loader)
    
    if verbose :
        print(f"Number of training epochs & steps: {total_epochs} {n_steps}")

    ##############

    all_metrics = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["train"] = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["test"] = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["steps_epoch"] = {}

    ##############

    train_statistics = eval_model(model, train_loader_for_eval, device)
    for k, v in train_statistics.items() :
        all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device) 
    for k, v in test_statistics.items() :
        all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(0)
    all_metrics["steps_epoch"][0] = 0


    ######################
    # Save model
    state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, f"{checkpoint_path}/{exp_name}_state_{0}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
  
    
    ##############

    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    if verbose :
        to_print = "\n" + " | ".join(f"Train {k} : {v:.6f}" for k, v in train_statistics.items())
        to_print += " | " + " | ".join(f"Test {k} : {v:.6f}" for k, v in test_statistics.items())
        to_print += f" | lr = {current_lr}"
        print(to_print)

    ##############

    cur_step = 1 
    tol_step = 0

    for epoch in tqdm(range(1, total_epochs+1), desc="Training", total=total_epochs):

        # start_time = time.time()
        
        for i, batch in enumerate(train_loader) :
            batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            model.train()

            logits, *_ = model(batch_x) # (B, S, V)
            loss, _ = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ==========================
            # TODO: Write your code here
            # ==========================
            # scheduler.step()
            # current_lr = scheduler.optimizer.param_groups[0]["lr"]
            # ==========================
            # ==========================
              
            if cur_step in [1, n_steps] or cur_step % eval_period == 0 or cur_step <= eval_first:
                train_statistics = eval_model(model, train_loader_for_eval, device)
                for k, v in train_statistics.items() : all_metrics["train"][k].append(v)

                test_statistics = eval_model(model, test_loader, device)
                for k, v in test_statistics.items() : all_metrics["test"][k].append(v)

                all_metrics["all_steps"].append(cur_step)
                all_metrics["steps_epoch"][cur_step] = epoch

            
            if  verbose and (cur_step in [1, n_steps] or cur_step%print_step==0) :
                to_print = "\n" + " | ".join(f"Train {k} : {v:.6f}" for k, v in train_statistics.items())
                to_print += " | " + " | ".join(f"Test {k} : {v:.6f}" for k, v in test_statistics.items())
                to_print += f" | lr = {current_lr}"
                print(to_print)

            if cur_step in [1, n_steps] or cur_step%save_model_step==0 or cur_step <= eval_first : 
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
                

            if cur_step in [1, n_steps] or cur_step%save_statistic_step==0:
                #to_save = {k:v for k, v in all_metrics.items()}
                to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()} # to avoid issues with lambda
                torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

            cur_step += 1

        # ==========================
        # TODO: Write your code here
        # ==========================
        ###
        # scheduler.step() 
        # current_lr = scheduler.optimizer.param_groups[0]["lr"]
        # ==========================
        # ==========================

        ##############
        # You can implement early stopping here.
        # That is, if the model does not improve for a certain number of steps, you can stop the training.
        ##############

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time for one step : {elapsed_time} seconds")

    state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
    
    train_statistics = eval_model(model, train_loader_for_eval, device)
    for k, v in train_statistics.items() : all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device)
    for k, v in test_statistics.items() : all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(cur_step)
    all_metrics["steps_epoch"][cur_step] = epoch

    to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()} # to avoid issues with lambda
    torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

    return all_metrics
