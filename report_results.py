import os
import pandas as pd
import torch
from checkpointing import get_extrema_performance_steps_per_trials
import numpy as np
import argparse
from pathlib import Path
from plotter import plot_loss_accuracy, plot_comparative_performance, plot_loss_accuracy_q3_a, plot_loss_accuracy_q3_b, plot_loss_accuracy_q4, plot_loss_accuracy_q5_a, plot_loss_accuracy_q5_b, plot_loss_accuracy_q6_a, plot_loss_accuracy_q6_b

def fetch_metrics(log_dir):
    """Recursively find all test.pth files in log_dir and collect metrics"""
    metrics_dict = {
        "train": {
            "loss": [],
            "accuracy": []
        },
        "test": {
            "loss": [],
            "accuracy": []
        },
        "all_steps": []
    }
    
    # Recursively find all test.pth files
    for test_file in Path(log_dir).rglob('test.pth'):
        # Load the all_metrics dictionary from the checkpoint file
        all_metrics = torch.load(test_file, map_location='cpu')
        print(test_file)
        
        # Append the metrics to the dictionary
        metrics_dict["all_steps"].append(all_metrics["all_steps"])
        metrics_dict["train"]["loss"].append(all_metrics["train"]["loss"])
        metrics_dict["train"]["accuracy"].append(all_metrics["train"]["accuracy"])
        metrics_dict["test"]["loss"].append(all_metrics["test"]["loss"])
        metrics_dict["test"]["accuracy"].append(all_metrics["test"]["accuracy"])

    return metrics_dict

def fetch_metrics_for_parameter(model_dir, parameter_value):
    """Fetch metrics for a specific r_train value"""
    metrics_dict = {
        "train": {
            "loss": [],
            "accuracy": []
        },
        "test": {
            "loss": [],
            "accuracy": []
        },
        "l2_norm": [],
        "all_steps": []
    }
    
    r_dir = model_dir / str(parameter_value)
    # Look in seed directories (0 and 42)
    for seed_dir in r_dir.iterdir():
        if seed_dir.is_dir():
            test_file = seed_dir / 'test.pth'
            if test_file.exists():
                all_metrics = torch.load(test_file, map_location='cpu')
                print(f"Loading {test_file}")
                
                metrics_dict["all_steps"].append(all_metrics["all_steps"])
                metrics_dict["train"]["loss"].append(all_metrics["train"]["loss"])
                metrics_dict["train"]["accuracy"].append(all_metrics["train"]["accuracy"])
                metrics_dict["test"]["loss"].append(all_metrics["test"]["loss"])
                metrics_dict["test"]["accuracy"].append(all_metrics["test"]["accuracy"])
                metrics_dict["l2_norm"].append(all_metrics["l2_norm"])
    return metrics_dict


def q1(log_dir, results_dir):
    """Original implementation for Q1 and Q2"""
    # Get all model directories in logs/{question} 
    models = [d.name for d in log_dir.iterdir() if d.is_dir()]
    results = []

    for model in models:
        metrics = fetch_metrics(log_dir / model)
        metrics_summary = get_extrema_performance_steps_per_trials(metrics)
        
        result = {
            "Model": model,
            "L_train": f"{metrics_summary['min_train_loss']:.2e} ± {metrics_summary['min_train_loss_std']:.2e}",
            "L_val": f"{metrics_summary['min_test_loss']:.2e} ± {metrics_summary['min_test_loss_std']:.2e}",
            "A_train": f"{metrics_summary['max_train_accuracy']:.2e} ± {metrics_summary['max_train_accuracy_std']:.2e}",
            "A_val": f"{metrics_summary['max_test_accuracy']:.2e} ± {metrics_summary['max_test_accuracy_std']:.2e}",
            "tf(L_train)": f"{metrics_summary['min_train_loss_step']:.2f}",
            "tf(L_val)": f"{metrics_summary['min_test_loss_step']:.2f}",
            "tf(A_train)": f"{metrics_summary['max_train_accuracy_step']:.2f}",
            "tf(A_val)": f"{metrics_summary['max_test_accuracy_step']:.2f}",
            "dt(L)": f"{metrics_summary['min_test_loss_step'] - metrics_summary['min_train_loss_step']:.2f}",
            "dt(A)": f"{metrics_summary['max_test_accuracy_step'] - metrics_summary['max_train_accuracy_step']:.2f}"
        }

        results.append(result)

        # Plot loss and accuracy for each model
        plot_loss_accuracy(metrics, model, results_dir)

    # After collecting all results, plot comparative performances
    plot_comparative_performance(results, results_dir)

    df = pd.DataFrame(results)
    return df

def q3(log_dir, results_dir):
    """Q3-specific implementation"""
    models = [d.name for d in log_dir.iterdir() if d.is_dir()]
    r_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    for model in models:
        model_dir = log_dir / model
        metrics_per_r = {}
        model_results = []
        
        # Collect metrics for each r value
        for r in r_values:
            metrics_per_r[r] = fetch_metrics_for_parameter(model_dir, r)
        
        # Plot with all r values
        plot_loss_accuracy_q3_a(metrics_per_r, model, results_dir / "a")
        
        # Calculate summary statistics for CSV
        for r in r_values:
            metrics = metrics_per_r[r]
            metrics_summary = get_extrema_performance_steps_per_trials(metrics)
            
            result = {
                "Model": model,
                "r_train": r,
                "L_train_min": metrics_summary['min_train_loss'],
                "L_val_min": metrics_summary['min_test_loss'],
                "A_train_max": metrics_summary['max_train_accuracy'],
                "A_val_max": metrics_summary['max_test_accuracy'],
                "L_train": f"{metrics_summary['min_train_loss']:.2e} ± {metrics_summary['min_train_loss_std']:.2e}",
                "L_val": f"{metrics_summary['min_test_loss']:.2e} ± {metrics_summary['min_test_loss_std']:.2e}",
                "A_train": f"{metrics_summary['max_train_accuracy']:.2e} ± {metrics_summary['max_train_accuracy_std']:.2e}",
                "A_val": f"{metrics_summary['max_test_accuracy']:.2e} ± {metrics_summary['max_test_accuracy_std']:.2e}",
                "tf(L_train)": f"{metrics_summary['min_train_loss_step']:.2f}",
                "tf(L_val)": f"{metrics_summary['min_test_loss_step']:.2f}",
                "tf(A_train)": f"{metrics_summary['max_train_accuracy_step']:.2f}",
                "tf(A_val)": f"{metrics_summary['max_test_accuracy_step']:.2f}",
                "dt(L)": f"{metrics_summary['min_test_loss_step'] - metrics_summary['min_train_loss_step']:.2f}",
                "dt(A)": f"{metrics_summary['max_test_accuracy_step'] - metrics_summary['max_train_accuracy_step']:.2f}"
            }
            model_results.append(result)

        results.extend(model_results)
        plot_loss_accuracy_q3_b(model_results, model, results_dir / "b")

    df = pd.DataFrame(results)
    return df

def q4(log_dir, results_dir):
    """Q4-specific implementation"""
    models = [d.name for d in log_dir.iterdir() if d.is_dir()]
    results = []
    operation_orders = [2, 3]

    for model in models:
        model_dir = log_dir / model
        metrics_per_order = {}

        for operation_order in operation_orders:
            metrics_per_order[operation_order] = fetch_metrics(model_dir / str(operation_order))

        plot_loss_accuracy_q4(metrics_per_order, model, results_dir)

        for operation_order in operation_orders:
            metrics = metrics_per_order[operation_order]
            metrics_summary = get_extrema_performance_steps_per_trials(metrics)

            result = {
                "Model": model,
                "operation_order": operation_order,
                "L_train": f"{metrics_summary['min_train_loss']:.2e} ± {metrics_summary['min_train_loss_std']:.2e}",
                "L_val": f"{metrics_summary['min_test_loss']:.2e} ± {metrics_summary['min_test_loss_std']:.2e}",
                "A_train": f"{metrics_summary['max_train_accuracy']:.2e} ± {metrics_summary['max_train_accuracy_std']:.2e}",
                "A_val": f"{metrics_summary['max_test_accuracy']:.2e} ± {metrics_summary['max_test_accuracy_std']:.2e}",
                "tf(L_train)": f"{metrics_summary['min_train_loss_step']:.2f}",
                "tf(L_val)": f"{metrics_summary['min_test_loss_step']:.2f}",
                "tf(A_train)": f"{metrics_summary['max_train_accuracy_step']:.2f}",
                "tf(A_val)": f"{metrics_summary['max_test_accuracy_step']:.2f}",
                "dt(L)": f"{metrics_summary['min_test_loss_step'] - metrics_summary['min_train_loss_step']:.2f}",
                "dt(A)": f"{metrics_summary['max_test_accuracy_step'] - metrics_summary['max_train_accuracy_step']:.2f}"
            }
            results.append(result)

    df = pd.DataFrame(results)
    return df

def q5(log_dir, results_dir):
    """Q5-specific implementation"""
    models = [d.name for d in log_dir.iterdir() if d.is_dir()]
    results = []

    layer_vals = [1, 2, 3]
    embedding_sizes = [2**6, 2**7, 2**8]

    for model in models:
        model_dir = log_dir / model
        metrics_per_layer = {}
        metrics_per_embedding_size = {}
        model_results = []

        for layer in layer_vals:
            model_dir = model_dir / str(layer)
            for d in embedding_sizes:
                metrics_per_embedding_size[d] = fetch_metrics_for_parameter(model_dir, d)
                metrics_summary = get_extrema_performance_steps_per_trials(metrics_per_embedding_size[d])
                result = {
                    "Model": model,
                    "Layer": layer,
                    "Embedding Size": d,
                    "L_train_min": metrics_summary['min_train_loss'],
                    "L_val_min": metrics_summary['min_test_loss'],
                    "A_train_max": metrics_summary['max_train_accuracy'],
                    "A_val_max": metrics_summary['max_test_accuracy'],
                    "L_train": f"{metrics_summary['min_train_loss']:.2e} ± {metrics_summary['min_train_loss_std']:.2e}",
                    "L_val": f"{metrics_summary['min_test_loss']:.2e} ± {metrics_summary['min_test_loss_std']:.2e}",
                    "A_train": f"{metrics_summary['max_train_accuracy']:.2e} ± {metrics_summary['max_train_accuracy_std']:.2e}",
                    "A_val": f"{metrics_summary['max_test_accuracy']:.2e} ± {metrics_summary['max_test_accuracy_std']:.2e}",
                    "tf(L_train)": f"{metrics_summary['min_train_loss_step']:.2f}",
                    "tf(L_val)": f"{metrics_summary['min_test_loss_step']:.2f}",
                    "tf(A_train)": f"{metrics_summary['max_train_accuracy_step']:.2f}",
                    "tf(A_val)": f"{metrics_summary['max_test_accuracy_step']:.2f}",
                    "dt(L)": f"{metrics_summary['min_test_loss_step'] - metrics_summary['min_train_loss_step']:.2f}",
                    "dt(A)": f"{metrics_summary['max_test_accuracy_step'] - metrics_summary['max_train_accuracy_step']:.2f}"
                }
                model_results.append(result)
            
            metrics_per_layer[layer] = metrics_per_embedding_size
            results.extend(model_results)

        plot_loss_accuracy_q5_a(metrics_per_layer, model, results_dir / "a" )
        plot_loss_accuracy_q5_b(model_results, model, results_dir / "b")


    df = pd.DataFrame(results)
    return df

def q6(log_dir, results_dir):
    """Q6-specific implementation"""
    models = [d.name for d in log_dir.iterdir() if d.is_dir()]
    results = []

    batch_sizes = [2**5, 2**6, 2**7, 2**8, 2**9]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for model in models:
        metrics_per_batch_size = {}
        metrics_per_alpha = {}

        for batch_size in batch_sizes:
            model_dir = log_dir / model / str(batch_size)
            for alpha in alphas:
                metrics_per_alpha[alpha] = fetch_metrics_for_parameter(model_dir, alpha)
                metrics_summary = get_extrema_performance_steps_per_trials(metrics_per_alpha[alpha])
                result = {
                    "Model": model,
                    "Batch Size": batch_size,
                    "Alpha": alpha,
                    "L_train": f"{metrics_summary['min_train_loss']:.2e} ± {metrics_summary['min_train_loss_std']:.2e}",
                    "L_val": f"{metrics_summary['min_test_loss']:.2e} ± {metrics_summary['min_test_loss_std']:.2e}",
                    "A_train": f"{metrics_summary['max_train_accuracy']:.2e} ± {metrics_summary['max_train_accuracy_std']:.2e}",
                    "A_val": f"{metrics_summary['max_test_accuracy']:.2e} ± {metrics_summary['max_test_accuracy_std']:.2e}",
                    "tf(L_train)": f"{metrics_summary['min_train_loss_step']:.2f}",
                    "tf(L_val)": f"{metrics_summary['min_test_loss_step']:.2f}",
                    "tf(A_train)": f"{metrics_summary['max_train_accuracy_step']:.2f}",
                    "tf(A_val)": f"{metrics_summary['max_test_accuracy_step']:.2f}",
                    "dt(L)": f"{metrics_summary['min_test_loss_step'] - metrics_summary['min_train_loss_step']:.2f}",
                    "dt(A)": f"{metrics_summary['max_test_accuracy_step'] - metrics_summary['max_train_accuracy_step']:.2f}"
                }
                results.append(result)

            metrics_per_batch_size[batch_size] = metrics_per_alpha

        plot_loss_accuracy_q6_a(metrics_per_batch_size, model, results_dir / "a")
        plot_loss_accuracy_q6_b(metrics_per_batch_size, model, results_dir / "b")

    df = pd.DataFrame(results)
    return df

def q7(log_dir, results_dir):
    """Q7-specific implementation"""
    models = [d.name for d in log_dir.iterdir() if d.is_dir()]
    results = []

    weight_decays = [1/4, 1/2, 3/4, 1]

    for model in models:
        model_dir = log_dir / model
        metrics_per_weight_decay = {}

        for weight_decay in weight_decays:
            metrics_per_weight_decay[weight_decay] = fetch_metrics_for_parameter(model_dir, weight_decay)

def report_results(question):
    results_dir = Path(f'results/{question}')
    results_dir.mkdir(exist_ok=True)
    
    log_dir = Path(f'logs/{question}')
    
    if question == "Q1":
        df = q1(log_dir, results_dir)
    elif question == "Q3":
        df = q3(log_dir, results_dir)
    elif question == "Q4":
        df = q4(log_dir, results_dir)
    elif question == "Q5":
        df = q5(log_dir, results_dir)
    elif question == "Q6":
        df = q6(log_dir, results_dir)
    elif question == "Q7":
        df = q7(log_dir, results_dir)
        
    df.to_csv(results_dir / f'{question}_results.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str, required=True, help='Question number (e.g. Q1)')
    args = parser.parse_args()
    
    report_results(args.question)
