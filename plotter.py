import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIGSIZE = (12, 6)  # Adjusted figure size
LINEWIDTH = 2.0
FONTSIZE = 12
def plot_loss_accs(
    statistics, multiple_runs=False, log_x=False, log_y=False, 
    figsize=FIGSIZE, linewidth=LINEWIDTH, fontsize=FONTSIZE,
    fileName=None, filePath=None, show=True
    ):

    rows, cols = 1, 2
    fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))
    color_1 = 'tab:blue' # #1f77b4
    color_2 = 'tab:red' # #d62728
    
    same_steps = False
    if multiple_runs :
        all_steps = statistics["all_steps"]
        same_steps = all(len(steps) == len(all_steps[0]) for steps in all_steps) # Check if all runs have the same number of steps
        if same_steps :
            all_steps = np.array(all_steps[0]) + 1e-0 # Add 1e-0 to avoid log(0)
        else :
            all_steps = [np.array(steps) + 1e-0 for steps in all_steps] # Add 1e-0 to avoid log(0)
            color_indices = np.linspace(0, 1, len(all_steps))
            colors = plt.cm.viridis(color_indices)
    else :
        all_steps = np.array(statistics["all_steps"]) + 1e-0

    for i, key in enumerate(["accuracy", "loss"]) :
        ax = fig.add_subplot(rows, cols, i+1)
        if multiple_runs :
            zs = np.array(statistics["train"][key])
            if same_steps :
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                #ax.errorbar(all_steps, zs_mean, yerr=zs_std, fmt=f'-', color=color_1, label=f"Train", lw=linewidth)
                ax.plot(all_steps, zs_mean, '-', color=color_1, label=f"train", lw=linewidth)
                ax.fill_between(all_steps, zs_mean-zs_std, zs_mean+zs_std, color=color_1, alpha=0.2)
            else :  
                for j, z in enumerate(zs) :
                    ax.plot(all_steps[j], z, '-', color=colors[j], label=f"train", lw=linewidth/2)

            zs = np.array(statistics["test"][key])
            if same_steps :
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                ax.plot(all_steps, zs_mean, '-', color=color_2, label=f"eval", lw=linewidth)
                ax.fill_between(all_steps, zs_mean-zs_std, test_mean+zs_std, color=color_2, alpha=0.2)
            else :  
                for j, z in enumerate(zs) :
                    ax.plot(all_steps[j], z, '--', color=colors[j], label=f"eval", lw=linewidth/2)

        else :
            ax.plot(all_steps, statistics["train"][key], "-", color=color_1,  label=f"train", lw=linewidth) 
            ax.plot(all_steps, statistics["test"][key], "-", color=color_2,  label=f"eval", lw=linewidth) 

        if log_x : ax.set_xscale('log')
        #if log_y : ax.set_yscale('log')
        if log_y and key=="loss" : ax.set_yscale('log') # No need to log accuracy
        ax.tick_params(axis='y', labelsize='x-large')
        ax.tick_params(axis='x', labelsize='x-large')
        ax.set_xlabel("Training Steps (t)", fontsize=fontsize)
        if key=="accuracy": s = "Accuracy"
        if key=="loss": s = "Loss"
        #ax.set_ylabel(s, fontsize=fontsize)
        ax.set_title(s, fontsize=fontsize)
        ax.grid(True)
        if multiple_runs and (not same_steps) :
            legend_elements = [Line2D([0], [0], color='k', lw=linewidth, linestyle='-', label='train'),
                            Line2D([0], [0], color='k', lw=linewidth, linestyle='--', label='eval')]
            ax.legend(handles=legend_elements, fontsize=fontsize)
        else :
            ax.legend(fontsize=fontsize)

    if fileName is not None and filePath is not None :
        os.makedirs(filePath, exist_ok=True)
        plt.savefig(f"{filePath}/{fileName}"  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

    if show : plt.show()
    else : plt.close()

def plot_loss_accuracy(metrics, model_name, results_dir):
    """Plot training and validation loss and accuracy as a function of training steps."""
    fig = plt.figure(figsize=(12, 6))
    color_1 = 'tab:blue'
    color_2 = 'tab:red'
    
    # Convert lists to numpy arrays for easier manipulation
    all_steps = np.array(metrics['all_steps'][0])  # Assuming all runs have same steps
    
    # Plot Loss
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Calculate mean and std for train loss
    train_losses = np.array(metrics['train']['loss'])
    train_mean = np.mean(train_losses, axis=0)
    train_std = np.std(train_losses, axis=0)
    
    # Calculate mean and std for test loss
    test_losses = np.array(metrics['test']['loss'])
    test_mean = np.mean(test_losses, axis=0)
    test_std = np.std(test_losses, axis=0)
    
    # Plot loss with fill_between for standard deviation
    ax1.plot(all_steps, train_mean, '-', color=color_1, label='train', lw=2.0)
    ax1.fill_between(all_steps, train_mean-train_std, train_mean+train_std, color=color_1, alpha=0.2)
    
    ax1.plot(all_steps, test_mean, '-', color=color_2, label='eval', lw=2.0)
    ax1.fill_between(all_steps, test_mean-test_std, test_mean+test_std, color=color_2, alpha=0.2)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Training Steps (t)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Loss vs Training Steps for {model_name}', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    
    # Plot Accuracy
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Calculate mean and std for train accuracy
    train_accs = np.array(metrics['train']['accuracy'])
    train_mean = np.mean(train_accs, axis=0)
    train_std = np.std(train_accs, axis=0)
    
    # Calculate mean and std for test accuracy
    test_accs = np.array(metrics['test']['accuracy'])
    test_mean = np.mean(test_accs, axis=0)
    test_std = np.std(test_accs, axis=0)
    
    # Plot accuracy with fill_between for standard deviation
    ax2.plot(all_steps, train_mean, '-', color=color_1, label='train', lw=2.0)
    ax2.fill_between(all_steps, train_mean-train_std, train_mean+train_std, color=color_1, alpha=0.2)
    
    ax2.plot(all_steps, test_mean, '-', color=color_2, label='eval', lw=2.0)
    ax2.fill_between(all_steps, test_mean-test_std, test_mean+test_std, color=color_2, alpha=0.2)
    
    ax2.set_xlabel('Training Steps (t)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Accuracy vs Training Steps for {model_name}', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparative_performance(results, results_dir):
    """Plot comparative performances of models as a function of r_train."""
    r_train_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    L_train = []
    L_val = []
    A_train = []
    A_val = []

    for result in results:
        L_train.append(result['L_train'])
        L_val.append(result['L_val'])
        A_train.append(result['A_train'])
        A_val.append(result['A_val'])

    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(r_train_values, L_train, label='train', marker='o')
    plt.plot(r_train_values, L_val, label='eval', marker='o')
    plt.yscale('log')  # Log scale for loss
    plt.xlabel('r_train')
    plt.ylabel('Loss')
    plt.title('Comparative Loss Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / 'comparative_loss_performance.png')
    plt.close()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(r_train_values, A_train, label='train', marker='o')
    plt.plot(r_train_values, A_val, label='eval', marker='o')
    plt.xlabel('r_train')
    plt.ylabel('Accuracy')
    plt.title('Comparative Accuracy Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / 'comparative_accuracy_performance.png')
    plt.close()

def plot_loss_accuracy_q3_a(metrics_per_r, model_name, results_dir):
    """Plot training and validation curves for Q3, with separate figures for each metric."""
    results_dir.mkdir(parents=True, exist_ok=True)
    width = 12  # Keep the same width for all figures
    height = 6
    
    # Create color map for different r values
    r_values = sorted(metrics_per_r.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(r_values)))
    
    q3_dir = results_dir
    
    # Plot Training Loss
    fig1 = plt.figure(figsize=(width, height))
    ax1 = fig1.add_subplot(111)
    
    for r, color in zip(r_values, colors):
        metrics = metrics_per_r[r]
        all_steps = np.array(metrics['all_steps'][0])
        
        train_losses = np.array(metrics['train']['loss'])
        train_mean = np.mean(train_losses, axis=0)
        train_std = np.std(train_losses, axis=0)
        
        ax1.plot(all_steps, train_mean, '-', color=color, label=f'r={r}', lw=2.0)
        ax1.fill_between(all_steps, train_mean-train_std, train_mean+train_std, color=color, alpha=0.2)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Training Steps (t)', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title(f'Training Loss vs Steps for {model_name.upper()}', fontsize=12)
    ax1.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(q3_dir / f'{model_name.upper()}_Q3_train_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Validation Loss
    fig2 = plt.figure(figsize=(width, height))
    ax2 = fig2.add_subplot(111)
    
    for r, color in zip(r_values, colors):
        metrics = metrics_per_r[r]
        all_steps = np.array(metrics['all_steps'][0])
        
        test_losses = np.array(metrics['test']['loss'])
        test_mean = np.mean(test_losses, axis=0)
        test_std = np.std(test_losses, axis=0)
        
        ax2.plot(all_steps, test_mean, '-', color=color, label=f'r={r}', lw=2.0)
        ax2.fill_between(all_steps, test_mean-test_std, test_mean+test_std, color=color, alpha=0.2)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Training Steps (t)', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title(f'Validation Loss vs Steps for {model_name.upper()}', fontsize=12)
    ax2.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(q3_dir / f'{model_name.upper()}_Q3_val_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Training Accuracy
    fig3 = plt.figure(figsize=(width, height))
    ax3 = fig3.add_subplot(111)
    
    for r, color in zip(r_values, colors):
        metrics = metrics_per_r[r]
        all_steps = np.array(metrics['all_steps'][0])
        
        train_accs = np.array(metrics['train']['accuracy'])
        train_mean = np.mean(train_accs, axis=0)
        train_std = np.std(train_accs, axis=0)
        
        ax3.plot(all_steps, train_mean, '-', color=color, label=f'r={r}', lw=2.0)
        ax3.fill_between(all_steps, train_mean-train_std, train_mean+train_std, color=color, alpha=0.2)
    
    ax3.set_xlabel('Training Steps (t)', fontsize=12)
    ax3.set_ylabel('Training Accuracy', fontsize=12)
    ax3.set_title(f'Training Accuracy vs Steps for {model_name.upper()}', fontsize=12)
    ax3.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig(q3_dir / f'{model_name.upper()}_Q3_train_acc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Validation Accuracy
    fig4 = plt.figure(figsize=(width, height))
    ax4 = fig4.add_subplot(111)
    
    for r, color in zip(r_values, colors):
        metrics = metrics_per_r[r]
        all_steps = np.array(metrics['all_steps'][0])
        
        test_accs = np.array(metrics['test']['accuracy'])
        test_mean = np.mean(test_accs, axis=0)
        test_std = np.std(test_accs, axis=0)
        
        ax4.plot(all_steps, test_mean, '-', color=color, label=f'r={r}', lw=2.0)
        ax4.fill_between(all_steps, test_mean-test_std, test_mean+test_std, color=color, alpha=0.2)
    
    ax4.set_xlabel('Training Steps (t)', fontsize=12)
    ax4.set_ylabel('Validation Accuracy', fontsize=12)
    ax4.set_title(f'Validation Accuracy vs Steps for {model_name.upper()}', fontsize=12)
    ax4.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True)
    plt.tight_layout()
    plt.savefig(q3_dir / f'{model_name.upper()}_Q3_val_acc.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_accuracy_q3_b(results, model_name, results_dir):
    """Plot training and validation curves for Q3, with separate figures for each metric, as a functino of r_train."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    r_train_values = [result['r_train'] for result in results]
    L_train_values = [float(result['L_train_min']) for result in results]  # Extracting the mean loss
    L_val_values = [float(result['L_val_min']) for result in results]  # Extracting the mean loss
    A_train_values = [float(result['A_train_max']) for result in results]  # Extracting the mean loss
    A_val_values = [float(result['A_val_max']) for result in results]  # Extracting the mean loss
    tf_L_train_values = [float(result['tf(L_train)']) for result in results]
    tf_L_val_values = [float(result['tf(L_val)']) for result in results]
    tf_A_train_values = [float(result['tf(A_train)']) for result in results]
    tf_A_val_values = [float(result['tf(A_val)']) for result in results]

    colors = plt.cm.viridis(np.linspace(0, 1, len(r_train_values)))

    # Plot Training and Validation Loss
    plt.figure(figsize=FIGSIZE)
    plt.plot(r_train_values, L_train_values, marker='o', linestyle='-', color='b', label='train')
    plt.plot(r_train_values, L_val_values, marker='o', linestyle='-', color='r', label='eval')
    plt.xlabel('r_train')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. r_train for {model_name.upper()}')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_loss_vs_r_train.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Training and Validation Accuracy
    plt.figure(figsize=FIGSIZE)
    plt.plot(r_train_values, A_train_values, marker='o', linestyle='-', color='g', label='train')
    plt.plot(r_train_values, A_val_values, marker='o', linestyle='-', color='y', label='eval')
    plt.xlabel('r_train')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. r_train for {model_name.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_accuracy_vs_r_train.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Training and Validation tf values
    plt.figure(figsize=FIGSIZE)
    plt.plot(r_train_values, tf_L_train_values, marker='o', linestyle='-', color='c', label='train')
    plt.plot(r_train_values, tf_L_val_values, marker='o', linestyle='-', color='m', label='eval')
    plt.xlabel('r_train')
    plt.ylabel('tf(L)')
    plt.title(f'tf(L) Values vs. r_train for {model_name.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_tf(L)_vs_r_train.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Training and Validation tf(A) values
    plt.figure(figsize=FIGSIZE)
    plt.plot(r_train_values, tf_A_train_values, marker='o', linestyle='-', color='c', label='train')
    plt.plot(r_train_values, tf_A_val_values, marker='o', linestyle='-', color='m', label='eval')
    plt.xlabel('r_train')
    plt.ylabel('tf(A)')
    plt.title(f'tf(A) Values vs. r_train for {model_name.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_tf(A)_vs_r_train.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_accuracy_q4(metrics_per_order, model_name, results_dir):
    """Plot training and validation curves for Q3, with separate figures for each metric."""
    width = 12  # Keep the same width for all figures
    height = 6
    
    # Create color map for different r values
    order_values = sorted(metrics_per_order.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(order_values)))
        
    # Plot Training Loss
    fig1 = plt.figure(figsize=(width, height))
    ax1 = fig1.add_subplot(111)
    
    for order, color in zip(order_values, colors):
        metrics = metrics_per_order[order]
        all_steps = np.array(metrics['all_steps'][0])
        
        train_losses = np.array(metrics['train']['loss'])
        train_mean = np.mean(train_losses, axis=0)
        train_std = np.std(train_losses, axis=0)
        
        ax1.plot(all_steps, train_mean, '-', color=color, label=f'operation order = {order}', lw=2.0)
        ax1.fill_between(all_steps, train_mean-train_std, train_mean+train_std, color=color, alpha=0.2)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Training Steps (t)', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title(f'Training Loss vs Steps for {model_name.upper()}', fontsize=12)
    ax1.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q4_train_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Validation Loss
    fig2 = plt.figure(figsize=(width, height))
    ax2 = fig2.add_subplot(111)
    
    for order, color in zip(order_values, colors):
        metrics = metrics_per_order[order]
        all_steps = np.array(metrics['all_steps'][0])
        
        test_losses = np.array(metrics['test']['loss'])
        test_mean = np.mean(test_losses, axis=0)
        test_std = np.std(test_losses, axis=0)
        
        ax2.plot(all_steps, test_mean, '-', color=color, label=f'operation order = {order}', lw=2.0)
        ax2.fill_between(all_steps, test_mean-test_std, test_mean+test_std, color=color, alpha=0.2)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Training Steps (t)', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title(f'Validation Loss vs Steps for {model_name.upper()}', fontsize=12)
    ax2.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q4_eval_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Training Accuracy
    fig3 = plt.figure(figsize=(width, height))
    ax3 = fig3.add_subplot(111)
    
    for order, color in zip(order_values, colors):
        metrics = metrics_per_order[order]
        all_steps = np.array(metrics['all_steps'][0])
        
        train_accs = np.array(metrics['train']['accuracy'])
        train_mean = np.mean(train_accs, axis=0)
        train_std = np.std(train_accs, axis=0)
        
        ax3.plot(all_steps, train_mean, '-', color=color, label=f'operation order = {order}', lw=2.0)
        ax3.fill_between(all_steps, train_mean-train_std, train_mean+train_std, color=color, alpha=0.2)
    
    ax3.set_xlabel('Training Steps (t)', fontsize=12)
    ax3.set_ylabel('Training Accuracy', fontsize=12)
    ax3.set_title(f'Training Accuracy vs Steps for {model_name.upper()}', fontsize=12)
    ax3.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q4_train_acc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Validation Accuracy
    fig4 = plt.figure(figsize=(width, height))
    ax4 = fig4.add_subplot(111)
    
    for order, color in zip(order_values, colors):
        metrics = metrics_per_order[order]
        all_steps = np.array(metrics['all_steps'][0])
        
        test_accs = np.array(metrics['test']['accuracy'])
        test_mean = np.mean(test_accs, axis=0)
        test_std = np.std(test_accs, axis=0)
        
        ax4.plot(all_steps, test_mean, '-', color=color, label=f'operation order = {order}', lw=2.0)
        ax4.fill_between(all_steps, test_mean-test_std, test_mean+test_std, color=color, alpha=0.2)
    
    ax4.set_xlabel('Training Steps (t)', fontsize=12)
    ax4.set_ylabel('Validation Accuracy', fontsize=12)
    ax4.set_title(f'Validation Accuracy vs Steps for {model_name.upper()}', fontsize=12)
    ax4.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q4_eval_acc.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_accuracy_q5_a(metrics_per_layer, model_name, results_dir):
    """Plot training and validation curves for Q5, with separate figures for each layer."""
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    width = 12  # Keep the same width for all figures
    height = 6
    
    # Create color map for different embedding sizes
    layer_vals = sorted(metrics_per_layer.keys())
    
    # Get all embedding sizes from first layer's metrics
    embedding_sizes = sorted(metrics_per_layer[layer_vals[0]].keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(embedding_sizes)))
    
    # For each layer value, create 4 plots (train/val loss/acc)
    for layer in layer_vals:
        # Plot Training Loss
        fig1 = plt.figure(figsize=(width, height))
        ax1 = fig1.add_subplot(111)
        
        for d, color in zip(embedding_sizes, colors):
            metrics = metrics_per_layer[layer][d]
            all_steps = np.array(metrics['all_steps'][0])
            
            train_losses = np.array(metrics['train']['loss'])
            train_mean = np.mean(train_losses, axis=0)
            train_std = np.std(train_losses, axis=0)
            
            d_exp = int(np.log2(d))  # Convert d to its base-2 exponent
            ax1.plot(all_steps, train_mean, '-', color=color, label=f'd = 2^{d_exp}', lw=2.0)
            ax1.fill_between(all_steps, train_mean-train_std, train_mean+train_std, color=color, alpha=0.2)
        
        ax1.set_yscale('log')
        ax1.set_xlabel('Training Steps (t)', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title(f'Training Loss vs Steps for {model_name.upper()} (L={layer})', fontsize=12)
        ax1.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / f'{model_name.upper()}_Q5_L{layer}_train_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot Validation Loss
        fig2 = plt.figure(figsize=(width, height))
        ax2 = fig2.add_subplot(111)
        
        for d, color in zip(embedding_sizes, colors):
            metrics = metrics_per_layer[layer][d]
            all_steps = np.array(metrics['all_steps'][0])
            
            test_losses = np.array(metrics['test']['loss'])
            test_mean = np.mean(test_losses, axis=0)
            test_std = np.std(test_losses, axis=0)
            
            d_exp = int(np.log2(d))
            ax2.plot(all_steps, test_mean, '-', color=color, label=f'd = 2^{d_exp}', lw=2.0)
            ax2.fill_between(all_steps, test_mean-test_std, test_mean+test_std, color=color, alpha=0.2)
        
        ax2.set_yscale('log')
        ax2.set_xlabel('Training Steps (t)', fontsize=12)
        ax2.set_ylabel('Validation Loss', fontsize=12)
        ax2.set_title(f'Validation Loss vs Steps for {model_name.upper()} (L={layer})', fontsize=12)
        ax2.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / f'{model_name.upper()}_Q5_L{layer}_eval_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot Training Accuracy
        fig3 = plt.figure(figsize=(width, height))
        ax3 = fig3.add_subplot(111)
        
        for d, color in zip(embedding_sizes, colors):
            metrics = metrics_per_layer[layer][d]
            all_steps = np.array(metrics['all_steps'][0])
            
            train_accs = np.array(metrics['train']['accuracy'])
            train_mean = np.mean(train_accs, axis=0)
            train_std = np.std(train_accs, axis=0)
            
            d_exp = int(np.log2(d))
            ax3.plot(all_steps, train_mean, '-', color=color, label=f'd = 2^{d_exp}', lw=2.0)
            ax3.fill_between(all_steps, train_mean-train_std, train_mean+train_std, color=color, alpha=0.2)
        
        ax3.set_xlabel('Training Steps (t)', fontsize=12)
        ax3.set_ylabel('Training Accuracy', fontsize=12)
        ax3.set_title(f'Training Accuracy vs Steps for {model_name.upper()} (L={layer})', fontsize=12)
        ax3.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / f'{model_name.upper()}_Q5_L{layer}_train_acc.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot Validation Accuracy
        fig4 = plt.figure(figsize=(width, height))
        ax4 = fig4.add_subplot(111)
        
        for d, color in zip(embedding_sizes, colors):
            metrics = metrics_per_layer[layer][d]
            all_steps = np.array(metrics['all_steps'][0])
            
            test_accs = np.array(metrics['test']['accuracy'])
            test_mean = np.mean(test_accs, axis=0)
            test_std = np.std(test_accs, axis=0)
            
            d_exp = int(np.log2(d))
            ax4.plot(all_steps, test_mean, '-', color=color, label=f'd = 2^{d_exp}', lw=2.0)
            ax4.fill_between(all_steps, test_mean-test_std, test_mean+test_std, color=color, alpha=0.2)
        
        ax4.set_xlabel('Training Steps (t)', fontsize=12)
        ax4.set_ylabel('Validation Accuracy', fontsize=12)
        ax4.set_title(f'Validation Accuracy vs Steps for {model_name.upper()} (L={layer})', fontsize=12)
        ax4.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / f'{model_name.upper()}_Q5_L{layer}_eval_acc.png', dpi=300, bbox_inches='tight')
        plt.close()
        
def plot_loss_accuracy_q5_b(results, model_name, results_dir):
    """Plot training and validation curves for Q5, with separate figures for each metric, as a function of embedding size d."""
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Extracting values from results
    d_values = [result['Embedding Size'] for result in results]
    L_train_values = [float(result['L_train_min']) for result in results]
    L_eval_values = [float(result['L_val_min']) for result in results]
    A_train_values = [float(result['A_train_max']) for result in results]
    A_eval_values = [float(result['A_val_max']) for result in results]
    tf_L_train_values = [float(result['tf(L_train)']) for result in results]
    tf_L_eval_values = [float(result['tf(L_val)']) for result in results]
    tf_A_train_values = [float(result['tf(A_train)']) for result in results]
    tf_A_eval_values = [float(result['tf(A_val)']) for result in results]
    layer_values = [result['Layer'] for result in results]

    # Create a color map for layers
    unique_layers = sorted(set(layer_values))
    color_map = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))
    layer_color_dict = {layer: color for layer, color in zip(unique_layers, color_map)}

    # Plot Training Loss vs. d
    plt.figure(figsize=FIGSIZE)
    for layer in unique_layers:
        mask = [l == layer for l in layer_values]
        plt.plot(np.array(d_values)[mask], np.array(L_train_values)[mask], '-o', 
                color=layer_color_dict[layer], label=f'Layer {layer}', alpha=0.7)
    plt.xlabel('Embedding Size')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss vs. Embedding Size for {model_name.upper()}')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_L_train_vs_d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Validation Loss vs. d
    plt.figure(figsize=FIGSIZE)
    for layer in unique_layers:
        mask = [l == layer for l in layer_values]
        plt.plot(np.array(d_values)[mask], np.array(L_eval_values)[mask], '-o',
                color=layer_color_dict[layer], label=f'Layer {layer}', alpha=0.7)
    plt.xlabel('Embedding Size')
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss vs. Embedding Size for {model_name.upper()}')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_L_eval_vs_d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Training Accuracy vs. d
    plt.figure(figsize=FIGSIZE)
    for layer in unique_layers:
        mask = [l == layer for l in layer_values]
        plt.plot(np.array(d_values)[mask], np.array(A_train_values)[mask], '-o',
                color=layer_color_dict[layer], label=f'Layer {layer}', alpha=0.7)
    plt.xlabel('Embedding Size')
    plt.ylabel('Training Accuracy')
    plt.title(f'Training Accuracy vs. Embedding Size for {model_name.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_A_train_vs_d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Validation Accuracy vs. d
    plt.figure(figsize=FIGSIZE)
    for layer in unique_layers:
        mask = [l == layer for l in layer_values]
        plt.plot(np.array(d_values)[mask], np.array(A_eval_values)[mask], '-o',
                color=layer_color_dict[layer], label=f'Layer {layer}', alpha=0.7)
    plt.xlabel('Embedding Size')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Validation Accuracy vs. Embedding Size for {model_name.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_A_eval_vs_d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Training tf(L) vs. d
    plt.figure(figsize=FIGSIZE)
    for layer in unique_layers:
        mask = [l == layer for l in layer_values]
        plt.plot(np.array(d_values)[mask], np.array(tf_L_train_values)[mask], '-o',
                color=layer_color_dict[layer], label=f'Layer {layer}', alpha=0.7)
    plt.xlabel('Embedding Size')
    plt.ylabel('tf(L_train)')
    plt.title(f'tf(L_train) vs. Embedding Size for {model_name.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_tf_L_train_vs_d.png', dpi=300, bbox_inches='tight')
    plt.close()
        
    # Plot Validation tf(L) vs. d
    plt.figure(figsize=FIGSIZE)
    for layer in unique_layers:
        mask = [l == layer for l in layer_values]
        plt.plot(np.array(d_values)[mask], np.array(tf_L_eval_values)[mask], '-o',
                color=layer_color_dict[layer], label=f'Layer {layer}', alpha=0.7)
    plt.xlabel('Embedding Size')
    plt.ylabel('tf(L_eval)')
    plt.title(f'tf(L_eval) vs. Embedding Size for {model_name.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_tf_L_eval_vs_d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Training tf(A) vs. d
    plt.figure(figsize=FIGSIZE)
    for layer in unique_layers:
        mask = [l == layer for l in layer_values]
        plt.plot(np.array(d_values)[mask], np.array(tf_A_train_values)[mask], '-o',
                color=layer_color_dict[layer], label=f'Layer {layer}', alpha=0.7)
    plt.xlabel('Embedding Size')
    plt.ylabel('tf(A_train)')
    plt.title(f'tf(A_train) vs. Embedding Size for {model_name.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_tf_A_train_vs_d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Validation tf(A) vs. d
    plt.figure(figsize=FIGSIZE)
    for layer in unique_layers:
        mask = [l == layer for l in layer_values]
        plt.plot(np.array(d_values)[mask], np.array(tf_A_eval_values)[mask], '-o',
                color=layer_color_dict[layer], label=f'Layer {layer}', alpha=0.7)
    plt.xlabel('Embedding Size')
    plt.ylabel('tf(A_eval)')
    plt.title(f'tf(A_eval) vs. Embedding Size for {model_name.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_tf_A_eval_vs_d.png', dpi=300, bbox_inches='tight')
    plt.close()
        
    plot_loss_accuracy_q5_b_params(results, model_name, results_dir)

def plot_loss_accuracy_q5_b_params(results, model_name, results_dir):
    """Plot training and validation curves for Q5, with separate figures for each metric."""
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Original value extraction
    d_values = [result['Embedding Size'] for result in results]
    L_train_values = [float(result['L_train_min']) for result in results]
    L_eval_values = [float(result['L_val_min']) for result in results]
    A_train_values = [float(result['A_train_max']) for result in results]
    A_eval_values = [float(result['A_val_max']) for result in results]
    layer_values = [result['Layer'] for result in results]
    P_values = [result['Parameters'] for result in results]  # Get P values

    # Create a color map for layers
    unique_layers = sorted(set(layer_values))
    color_map = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))
    layer_color_dict = {layer: color for layer, color in zip(unique_layers, color_map)}
    
    # Plot Training and Validation Loss vs. P
    plt.figure(figsize=FIGSIZE)
    for layer in unique_layers:
        mask = [l == layer for l in layer_values]
        plt.plot(np.array(P_values)[mask], np.array(L_train_values)[mask], '-o',
                color=layer_color_dict[layer], label=f'Layer {layer} (train)', alpha=0.7)
        plt.plot(np.array(P_values)[mask], np.array(L_eval_values)[mask], '--s',
                color=layer_color_dict[layer], label=f'Layer {layer} (eval)', alpha=0.7)
    plt.xlabel('Number of Parameters')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. Parameters for {model_name.upper()}')
    plt.xscale('log')  # Log scale for parameters
    plt.yscale('log')  # Log scale for loss
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_loss_vs_params.png', dpi=300, bbox_inches='tight')
    plt.close()
        
    # Plot Training and Validation Accuracy vs. P
    plt.figure(figsize=FIGSIZE)
    for layer in unique_layers:
        mask = [l == layer for l in layer_values]
        plt.plot(np.array(P_values)[mask], np.array(A_train_values)[mask], '-o',
                color=layer_color_dict[layer], label=f'Layer {layer} (train)', alpha=0.7)
        plt.plot(np.array(P_values)[mask], np.array(A_eval_values)[mask], '--s',
                color=layer_color_dict[layer], label=f'Layer {layer} (eval)', alpha=0.7)
    plt.xlabel('Number of Parameters')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Parameters for {model_name.upper()}')
    plt.xscale('log')  # Log scale for parameters
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_accuracy_vs_params.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_accuracy_q6_a(metrics_per_batch_size, model_name, results_dir):
    """Plot training and validation curves for Q6, with all batch sizes on the same figures."""
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    width = 12  # Keep the same width for all figures
    height = 6
    
    # Create color map for different batch sizes
    batch_sizes = sorted(metrics_per_batch_size.keys())
    alphas = sorted(metrics_per_batch_size[batch_sizes[0]].keys())
    print(f"==== alphas: \n{alphas} =====")
    colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))

    # Convert alphas to alpha*T values
    T = 10000
    alpha_T_values = [alpha * T for alpha in alphas]

    # Plot Training Loss
    fig1 = plt.figure(figsize=(width, height))
    ax1 = fig1.add_subplot(111)
    
    for batch_size in batch_sizes:
        train_losses = [metrics_per_batch_size[batch_size][alpha]['min_train_loss'] for alpha in alphas]
        train_std = np.std(train_losses, axis=0)
        
        color_index = batch_sizes.index(batch_size)
        ax1.plot(alpha_T_values, train_losses, '-', color=colors[color_index], label=f'batch={batch_size}', lw=2.0)
       # ax1.fill_between(alpha_T_values, train_losses - train_std, train_losses + train_std, color=colors[color_index], alpha=0.2)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('αT', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title(f'Training Loss vs αT for {model_name.upper()}', fontsize=12)
    ax1.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q6_train_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Validation Loss
    fig2 = plt.figure(figsize=(width, height))
    ax2 = fig2.add_subplot(111)
    
    for batch_size in batch_sizes:
        eval_losses = [metrics_per_batch_size[batch_size][alpha]['min_test_loss'] for alpha in alphas]
        eval_std = np.std(eval_losses, axis=0)
        
        color_index = batch_sizes.index(batch_size)
        ax2.plot(alpha_T_values, eval_losses, '-', color=colors[color_index], label=f'batch={batch_size}', lw=2.0)
       # ax2.fill_between(alpha_T_values, eval_losses - eval_std, eval_losses + eval_std, color=colors[color_index], alpha=0.2)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('αT', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title(f'Validation Loss vs αT for {model_name.upper()}', fontsize=12)
    ax2.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q6_eval_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Training Accuracy
    fig3 = plt.figure(figsize=(width, height))
    ax3 = fig3.add_subplot(111)
    
    for batch_size in batch_sizes:
        train_accs = [metrics_per_batch_size[batch_size][alpha]['max_train_accuracy'] for alpha in alphas]
        train_std = np.std(train_accs, axis=0)
        
        color_index = batch_sizes.index(batch_size)
        ax3.plot(alpha_T_values, train_accs, '-', color=colors[color_index], label=f'batch={batch_size}', lw=2.0)
       # ax3.fill_between(alpha_T_values, train_accs - train_std, train_accs + train_std, color=colors[color_index], alpha=0.2)
    
    ax3.set_ylim([0.0, 1.02])
    ax3.set_xlabel('αT', fontsize=12)
    ax3.set_ylabel('Training Accuracy', fontsize=12)
    ax3.set_title(f'Training Accuracy vs αT for {model_name.upper()}', fontsize=12)
    ax3.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q6_train_acc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Validation Accuracy
    fig4 = plt.figure(figsize=(width, height))
    ax4 = fig4.add_subplot(111)
    
    for batch_size in batch_sizes:
        val_accs = [metrics_per_batch_size[batch_size][alpha]['max_test_accuracy'] for alpha in alphas]
        val_std = np.std(val_accs, axis=0)
        
        color_index = batch_sizes.index(batch_size)
        ax4.plot(alpha_T_values, val_accs, '-', color=colors[color_index], label=f'batch={batch_size}', lw=2.0)
       # ax4.fill_between(alpha_T_values, val_accs - val_std, val_accs + val_std, color=colors[color_index], alpha=0.2)
    
    ax4.set_ylim([0.0, 1.02])
    ax4.set_xlabel('αT', fontsize=12)
    ax4.set_ylabel('Validation Accuracy', fontsize=12)
    ax4.set_title(f'Validation Accuracy vs αT for {model_name.upper()}', fontsize=12)
    ax4.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q6_val_acc.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_accuracy_q6_b(metrics_per_batch_size, model_name, results_dir):
    """Plot training and validation curves for Q6, with batch sizes on the x-axis and different colors for each alpha."""
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    width = 12  # Keep the same width for all figures
    height = 6
    
    # Create color map for different alpha values
    batch_sizes = sorted(metrics_per_batch_size.keys())
    alphas = sorted(metrics_per_batch_size[batch_sizes[0]].keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

    # Convert batch sizes to a list for plotting
    batch_size_values = [int(bs) for bs in batch_sizes]

    # Plot Training Loss
    fig1, ax1 = plt.subplots(figsize=(width, height))
    for alpha, color in zip(alphas, colors):
        train_losses = [metrics_per_batch_size[batch_size][alpha]['min_train_loss'] for batch_size in batch_sizes]
        ax1.plot(batch_size_values, train_losses, '-o', color=color, label=f'α={alpha}', lw=2.0)

    ax1.set_yscale('log')
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title(f'Training Loss vs Batch Size for {model_name.upper()}', fontsize=12)
    ax1.legend(title='Alpha', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q6_train_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Validation Loss
    fig2, ax2 = plt.subplots(figsize=(width, height))
    for alpha, color in zip(alphas, colors):
        val_losses = [metrics_per_batch_size[batch_size][alpha]['min_test_loss'] for batch_size in batch_sizes]
        ax2.plot(batch_size_values, val_losses, '-o', color=color, label=f'α={alpha}', lw=2.0)

    ax2.set_yscale('log')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title(f'Validation Loss vs Batch Size for {model_name.upper()}', fontsize=12)
    ax2.legend(title='Alpha', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q6_val_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Training Accuracy
    fig3, ax3 = plt.subplots(figsize=(width, height))
    for alpha, color in zip(alphas, colors):
        train_accs = [metrics_per_batch_size[batch_size][alpha]['max_train_accuracy'] for batch_size in batch_sizes]
        ax3.plot(batch_size_values, train_accs, '-o', color=color, label=f'α={alpha}', lw=2.0)

    ax3.set_ylim([0.0, 1.02])
    ax3.set_xlabel('Batch Size', fontsize=12)
    ax3.set_ylabel('Training Accuracy', fontsize=12)
    ax3.set_title(f'Training Accuracy vs Batch Size for {model_name.upper()}', fontsize=12)
    ax3.legend(title='Alpha', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q6_train_acc.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Validation Accuracy
    fig4, ax4 = plt.subplots(figsize=(width, height))
    for alpha, color in zip(alphas, colors):
        val_accs = [metrics_per_batch_size[batch_size][alpha]['max_test_accuracy'] for batch_size in batch_sizes]
        ax4.plot(batch_size_values, val_accs, '-o', color=color, label=f'α={alpha}', lw=2.0)

    ax4.set_ylim([0.0, 1.02])
    ax4.set_xlabel('Batch Size', fontsize=12)
    ax4.set_ylabel('Validation Accuracy', fontsize=12)
    ax4.set_title(f'Validation Accuracy vs Batch Size for {model_name.upper()}', fontsize=12)
    ax4.legend(title='Alpha', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q6_val_acc.png', dpi=300, bbox_inches='tight')
    plt.close()