import os
import pandas as pd
import torch
from checkpointing import get_extrema_performance_steps_per_trials

def fetch_metrics(model):
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
    
    for seed in [0, 1]:
        log_dir = f'logs/Q1/{model}/{seed}'
        
        # Only load test.pth file
        path = os.path.join(log_dir, 'test.pth')
        if os.path.exists(path):
            # Load the all_metrics dictionary from the checkpoint file
            all_metrics = torch.load(path, map_location='cpu')
            
            # Append the metrics to the dictionary
            metrics_dict["all_steps"].append(all_metrics["all_steps"])
            metrics_dict["train"]["loss"].append(all_metrics["train"]["loss"])
            metrics_dict["train"]["accuracy"].append(all_metrics["train"]["accuracy"])
            metrics_dict["test"]["loss"].append(all_metrics["test"]["loss"])
            metrics_dict["test"]["accuracy"].append(all_metrics["test"]["accuracy"])

    return metrics_dict

def report_results():
    models = ["lstm", "gpt"]
    results = []

    for model in models:
        metrics = fetch_metrics(model)
        metrics_summary = get_extrema_performance_steps_per_trials(metrics)
        
        # Directly assign the best results
        best_train_loss = metrics_summary['min_train_loss']
        best_test_loss = metrics_summary['min_test_loss']
        best_train_accuracy = metrics_summary['max_train_accuracy']
        best_test_accuracy = metrics_summary['max_test_accuracy']

        result = {
            "Model": model,
            "L_train": f"{metrics_summary['min_train_loss']:.2e} ± {metrics_summary['min_train_loss_std']:.2e}",
            "L_val": f"{metrics_summary['min_test_loss']:.2e} ± {metrics_summary['min_test_loss_std']:.2e}",
            "A_train": f"{metrics_summary['max_train_accuracy']:.2e} ± {metrics_summary['max_train_accuracy_std']:.2e}",
            "A_val": f"{metrics_summary['max_test_accuracy']:.2e} ± {metrics_summary['max_test_accuracy_std']:.2e}",
            "tf(L_train)": metrics_summary['min_train_loss_step'],
            "tf(L_val)": metrics_summary['min_test_loss_step'],
            "tf(A_train)": metrics_summary['max_train_accuracy_step'],
            "tf(A_val)": metrics_summary['max_test_accuracy_step'],
            "Δt(L)": metrics_summary['min_test_loss_step'] - metrics_summary['min_train_loss_step'],
            "Δt(A)": metrics_summary['max_test_accuracy_step'] - metrics_summary['max_train_accuracy_step']
        }

        results.append(result)

    df = pd.DataFrame(results)
    
    # Clean up column names
    df.columns = [
        "Model",
        "Training Loss (L_train)",
        "Validation Loss (L_val)",
        "Training Accuracy (A_train)",
        "Validation Accuracy (A_val)",
        "Steps at L_train (tf(L_train))",
        "Steps at L_val (tf(L_val))",
        "Steps at A_train (tf(A_train))",
        "Steps at A_val (tf(A_val))",
        "Difference in Steps for Loss (delta t(L))",
        "Difference in Steps for Accuracy (delta t(A))"
    ]
    
    if not os.path.exists('results'):
        os.makedirs('results')
    df.to_csv('results/Q1_results.csv', index=False)

if __name__ == "__main__":
    report_results()
