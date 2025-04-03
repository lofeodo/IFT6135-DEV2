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
    """Plot training and validation loss and accuracy as a function of Training Steps (t)."""
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
    ax1.set_title(f'Loss vs Training Steps (t) for {model_name}', fontsize=12)
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
    ax2.set_title(f'Accuracy vs Training Steps (t) for {model_name}', fontsize=12)
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
    """Plot training and validation curves for Q3, with combined figures."""
    results_dir.mkdir(parents=True, exist_ok=True)
    width = 18
    height = 6
    
    # Create color map for different r values
    r_values = sorted(metrics_per_r.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(r_values)))
    
    q3_dir = results_dir
    
    # Plot Training Loss and Accuracy
    fig1 = plt.figure(figsize=(width, height))
    gs1 = fig1.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2)
    ax1_loss = fig1.add_subplot(gs1[0])
    ax1_acc = fig1.add_subplot(gs1[1])
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0.1, vmax=0.9))
    for r, color in zip(r_values, colors):
        metrics = metrics_per_r[r]
        all_steps = np.array(metrics['all_steps'][0])
        
        # Plot training loss
        train_losses = np.array(metrics['train']['loss'])
        train_loss_mean = np.mean(train_losses, axis=0)
        train_loss_std = np.std(train_losses, axis=0)
        
        ax1_loss.plot(all_steps, train_loss_mean, '-', color=color, lw=2.0)
        ax1_loss.fill_between(all_steps, train_loss_mean-train_loss_std, train_loss_mean+train_loss_std, color=color, alpha=0.2)
        
        # Plot training accuracy
        train_accs = np.array(metrics['train']['accuracy'])
        train_acc_mean = np.mean(train_accs, axis=0)
        train_acc_std = np.std(train_accs, axis=0)
        
        ax1_acc.plot(all_steps, train_acc_mean, '-', color=color, lw=2.0)
        ax1_acc.fill_between(all_steps, train_acc_mean-train_acc_std, train_acc_mean+train_acc_std, color=color, alpha=0.2)
    
    ax1_loss.set_yscale('log')
    ax1_loss.set_xlabel('Training Steps (t)', fontsize=12)
    ax1_loss.set_ylabel('Training Loss', fontsize=12)
    ax1_loss.grid(True)
    
    ax1_acc.set_xlabel('Training Steps (t)', fontsize=12)
    ax1_acc.set_ylabel('Training Accuracy', fontsize=12)
    ax1_acc.grid(True)
    
    plt.suptitle(f'Training Metrics vs Steps for {model_name.upper()}', fontsize=14)
    cbar = plt.colorbar(sm, ax=[ax1_loss, ax1_acc])
    cbar.set_label('r_train')
    plt.savefig(q3_dir / f'{model_name.upper()}_Q3_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Validation Loss and Accuracy
    fig2 = plt.figure(figsize=(width, height))
    gs2 = fig2.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2)
    ax2_loss = fig2.add_subplot(gs2[0])
    ax2_acc = fig2.add_subplot(gs2[1])
    
    for r, color in zip(r_values, colors):
        metrics = metrics_per_r[r]
        all_steps = np.array(metrics['all_steps'][0])
        
        # Plot validation loss
        val_losses = np.array(metrics['test']['loss'])
        val_loss_mean = np.mean(val_losses, axis=0)
        val_loss_std = np.std(val_losses, axis=0)
        
        ax2_loss.plot(all_steps, val_loss_mean, '-', color=color, lw=2.0)
        ax2_loss.fill_between(all_steps, val_loss_mean-val_loss_std, val_loss_mean+val_loss_std, color=color, alpha=0.2)
        
        # Plot validation accuracy
        val_accs = np.array(metrics['test']['accuracy'])
        val_acc_mean = np.mean(val_accs, axis=0)
        val_acc_std = np.std(val_accs, axis=0)
        
        ax2_acc.plot(all_steps, val_acc_mean, '-', color=color, lw=2.0)
        ax2_acc.fill_between(all_steps, val_acc_mean-val_acc_std, val_acc_mean+val_acc_std, color=color, alpha=0.2)
    
    ax2_loss.set_yscale('log')
    ax2_loss.set_xlabel('Training Steps (t)', fontsize=12)
    ax2_loss.set_ylabel('Validation Loss', fontsize=12)
    ax2_loss.grid(True)
    
    ax2_acc.set_xlabel('Training Steps (t)', fontsize=12)
    ax2_acc.set_ylabel('Validation Accuracy', fontsize=12)
    ax2_acc.grid(True)
    
    plt.suptitle(f'Validation Metrics vs Steps for {model_name.upper()}', fontsize=14)
    cbar = plt.colorbar(sm, ax=[ax2_loss, ax2_acc])
    cbar.set_label('r_train')
    plt.savefig(q3_dir / f'{model_name.upper()}_Q3_validation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_accuracy_q3_b(results, model_name, results_dir):
    """Plot training and validation curves for Q3, with combined figures."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    r_train_values = [result['r_train'] for result in results]
    L_train_values = [float(result['L_train_min']) for result in results]
    L_val_values = [float(result['L_val_min']) for result in results]
    A_train_values = [float(result['A_train_max']) for result in results]
    A_val_values = [float(result['A_val_max']) for result in results]
    tf_L_train_values = [float(result['tf(L_train)']) for result in results]
    tf_L_val_values = [float(result['tf(L_val)']) for result in results]
    tf_A_train_values = [float(result['tf(A_train)']) for result in results]
    tf_A_val_values = [float(result['tf(A_val)']) for result in results]

    # Create figure with three subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.25, wspace=0.1)
    ax1 = fig.add_subplot(gs[0])  # Loss
    ax2 = fig.add_subplot(gs[1])  # Accuracy
    ax3 = fig.add_subplot(gs[2])  # tf(L) values
    ax4 = fig.add_subplot(gs[3])  # tf(A) values
    
    # Plot loss
    ax1.plot(r_train_values, L_train_values, marker='o', linestyle='-', label='Train')
    ax1.plot(r_train_values, L_val_values, marker='s', linestyle='-', label='Validation')
    ax1.set_xlabel('r_train')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.set_xlim(0.1, 0.9)
    ax1.grid(True)
    ax1.set_title('Loss vs r_train')
    
    # Plot accuracy  
    ax2.plot(r_train_values, A_train_values, marker='o', linestyle='-', label='Train')
    ax2.plot(r_train_values, A_val_values, marker='s', linestyle='-', label='Validation')
    ax2.set_xlabel('r_train')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.02)
    ax2.set_xlim(0.1, 0.9)
    ax2.grid(True)
    ax2.set_title('Accuracy vs r_train')

    # Plot tf values
    ax3.plot(r_train_values, tf_L_train_values, marker='o', linestyle='-', label='tf(L) train')
    ax3.plot(r_train_values, tf_L_val_values, marker='s', linestyle='-', label='tf(L) eval')
    ax3.set_xlabel('r_train')
    ax3.set_ylabel('tf(L)')
    ax3.set_xlim(0.1, 0.9)
    ax3.set_ylim(0, 10010)
    ax3.grid(True)
    ax3.set_title('tf(L) Values vs r_train')

    ax4.plot(r_train_values, tf_A_train_values, marker='o', linestyle='-', label='tf(A) train')
    ax4.plot(r_train_values, tf_A_val_values, marker='s', linestyle='-', label='tf(A) eval')
    ax4.set_xlabel('r_train')
    ax4.set_ylabel('tf(A)')
    ax4.set_xlim(0.1, 0.9)
    ax4.set_ylim(0, 10010)
    ax4.grid(True)
    ax4.set_title('tf(A) Values vs r_train')

    # Add single legend for first two plots
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', label='Train', markersize=8),
        Line2D([0], [0], marker='s', color='gray', label='Validation', markersize=8)
    ]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.03, 0.5))
    
    # Add separate legend for tf plot
    plt.suptitle(f'Metrics vs r_train for {model_name.upper()}', fontsize=18)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_metrics_vs_r_train.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_accuracy_q4_a(metrics_per_order, model_name, results_dir):
    """Plot training and validation curves for Q4, with combined loss and accuracy plots for operation orders 2 and 3."""
    results_dir.mkdir(parents=True, exist_ok=True)
    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Create colormap for orders
    order_values = [2, 3]  # Only consider orders 2 and 3
    norm = plt.Normalize(2, 3)
    cmap = plt.cm.viridis

    # Create figure with extra space on right for colorbar and legend
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # Loss plot
    ax2 = fig.add_subplot(gs[0, 1])  # Accuracy plot
    
    # Initialize lists to store combined metrics
    combined_train_losses = []
    combined_test_losses = []
    combined_train_accs = []
    combined_test_accs = []
    all_steps = None

    # Plot loss and accuracy for each operation order
    for order in order_values:
        metrics = metrics_per_order[order]
        if all_steps is None:
            all_steps = np.array(metrics['all_steps'][0])  # Get steps from the first order
        
        # Calculate means and standard deviations
        train_losses = np.array(metrics['train']['loss'])
        test_losses = np.array(metrics['test']['loss'])
        train_accs = np.array(metrics['train']['accuracy'])
        test_accs = np.array(metrics['test']['accuracy'])
        
        # Combine metrics
        combined_train_losses.append(np.mean(train_losses, axis=0))
        combined_test_losses.append(np.mean(test_losses, axis=0))
        combined_train_accs.append(np.mean(train_accs, axis=0))
        combined_test_accs.append(np.mean(test_accs, axis=0))

    # Calculate overall means and standard deviations for combined metrics
    combined_train_loss_mean = np.mean(combined_train_losses, axis=0)
    combined_test_loss_mean = np.mean(combined_test_losses, axis=0)
    combined_train_acc_mean = np.mean(combined_train_accs, axis=0)
    combined_test_acc_mean = np.mean(combined_test_accs, axis=0)

    # Calculate standard deviations
    combined_train_loss_std = np.std(combined_train_losses, axis=0)
    combined_test_loss_std = np.std(combined_test_losses, axis=0)
    combined_train_acc_std = np.std(combined_train_accs, axis=0)
    combined_test_acc_std = np.std(combined_test_accs, axis=0)

    # Plot losses
    train_color = cmap(norm(2))  # Use color for order 2
    test_color = cmap(norm(3))  # Use color for order 3
    ax1.plot(all_steps, combined_train_loss_mean, '--', color=train_color, alpha=0.7, label='Train')
    ax1.fill_between(all_steps, combined_train_loss_mean - combined_train_loss_std, 
                     combined_train_loss_mean + combined_train_loss_std, color=train_color, alpha=0.1)
    ax1.plot(all_steps, combined_test_loss_mean, '-', color=test_color, alpha=0.7, label='Validation')
    ax1.fill_between(all_steps, combined_test_loss_mean - combined_test_loss_std, 
                     combined_test_loss_mean + combined_test_loss_std, color=test_color, alpha=0.1)

    # Plot accuracies
    ax2.plot(all_steps, combined_train_acc_mean, '--', color=train_color, alpha=0.7, label='Train')
    ax2.fill_between(all_steps, combined_train_acc_mean - combined_train_acc_std, 
                     combined_train_acc_mean + combined_train_acc_std, color=train_color, alpha=0.1)
    ax2.plot(all_steps, combined_test_acc_mean, '-', color=test_color, alpha=0.7, label='Validation')
    ax2.fill_between(all_steps, combined_test_acc_mean - combined_test_acc_std, 
                     combined_test_acc_mean + combined_test_acc_std, color=test_color, alpha=0.1)

    # Set axis labels and scales
    ax1.set_xlabel('Training Steps (t)', fontsize=12)
    ax2.set_xlabel('Training Steps (t)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    
    ax1.set_yscale('log')
    
    # Set axis limits
    ax1.set_xlim(0, 10000)
    ax2.set_xlim(0, 10000)
    ax2.set_ylim(0, 1.02)
    
    # Add titles and grid
    ax1.set_title('Loss vs Steps', fontsize=12)
    ax2.set_title('Accuracy vs Steps', fontsize=12)
    ax1.grid(True)
    ax2.grid(True)
    
    # Add legend
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    plt.suptitle(f'Metrics vs Steps for {model_name.upper()}', fontsize=12)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q4_metrics_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_loss_accuracy_q4_b(metrics_per_order, model_name, results_dir):
    """Plot training and validation curves for Q4, with separate loss and accuracy plots."""
    results_dir.mkdir(parents=True, exist_ok=True)
    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Create colormap for orders
    order_values = sorted(metrics_per_order.keys())
    norm = plt.Normalize(2, 3)
    cmap = plt.cm.viridis

        # Create figure with extra space on right for colorbar and legend
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    
    # Plot loss and accuracy for each operation order
    for order in order_values:
        metrics = metrics_per_order[order]
        all_steps = np.array(metrics['all_steps'][0])
        
        # Calculate means and standard deviations
        train_losses = np.array(metrics['train']['loss'])
        test_losses = np.array(metrics['test']['loss'])
        train_accs = np.array(metrics['train']['accuracy'])
        test_accs = np.array(metrics['test']['accuracy'])
        
        train_loss_mean = np.mean(train_losses, axis=0)
        test_loss_mean = np.mean(test_losses, axis=0)
        train_acc_mean = np.mean(train_accs, axis=0)
        test_acc_mean = np.mean(test_accs, axis=0)
        
        train_loss_std = np.std(train_losses, axis=0)
        test_loss_std = np.std(test_losses, axis=0)
        train_acc_std = np.std(train_accs, axis=0)
        test_acc_std = np.std(test_accs, axis=0)
        
        color = cmap(norm(order))
        
        # Plot losses
        ax1.plot(all_steps, train_loss_mean, '--', color=color, alpha=0.7)
        ax1.fill_between(all_steps, train_loss_mean-train_loss_std, train_loss_mean+train_loss_std, color=color, alpha=0.1)
        ax1.plot(all_steps, test_loss_mean, '-', color=color, alpha=0.7)
        ax1.fill_between(all_steps, test_loss_mean-test_loss_std, test_loss_mean+test_loss_std, color=color, alpha=0.1)
        
        # Plot accuracies
        ax2.plot(all_steps, train_acc_mean, '--', color=color, alpha=0.7)
        ax2.fill_between(all_steps, train_acc_mean-train_acc_std, train_acc_mean+train_acc_std, color=color, alpha=0.1)
        ax2.plot(all_steps, test_acc_mean, '-', color=color, alpha=0.7)
        ax2.fill_between(all_steps, test_acc_mean-test_acc_std, test_acc_mean+test_acc_std, color=color, alpha=0.1)
    
    # Set axis labels and scales
    ax1.set_xlabel('Training Steps (t)', fontsize=12)
    ax2.set_xlabel('Training Steps (t)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    
    ax1.set_yscale('log')
    
    # Set axis limits
    ax1.set_xlim(0, 10000)
    ax2.set_xlim(0, 10000)
    ax2.set_ylim(0, 1.02)
    
    # Add titles and grid
    ax1.set_title('Loss vs Steps', fontsize=12)
    ax2.set_title('Accuracy vs Steps', fontsize=12)
    ax1.grid(True)
    ax2.grid(True)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Order', ticks=[2, 3])
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], linestyle='--', color='gray', label='Train', markersize=8),
        Line2D([0], [0], linestyle='-', color='gray', label='Validation', markersize=8)
    ]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.0, 0.5))
    
    plt.suptitle(f'Metrics vs Steps for {model_name.upper()}', fontsize=12)
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name.upper()}_Q4_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_accuracy_q5_a(metrics_per_layer, model_name, results_dir):
    """Plot training and validation curves for Q5, with one combined plot per layer."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    layer_vals = sorted(metrics_per_layer.keys())
    embedding_sizes = sorted(metrics_per_layer[layer_vals[0]].keys())
    
    # Create normalization for the colorbars
    norm = plt.Normalize(min(np.log2(embedding_sizes)), max(np.log2(embedding_sizes)))
    
    for layer in layer_vals:
        # Plot losses and accuracies for each embedding size
        fig = plt.figure(figsize=(20, 6))
    
        # Create colormap for orders
        norm = plt.Normalize(min(np.log2(embedding_sizes)), max(np.log2(embedding_sizes)))
        cmap = plt.cm.viridis

            # Create figure with extra space on right for colorbar and legend
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        for d in embedding_sizes:
            metrics = metrics_per_layer[layer][d]
            all_steps = np.array(metrics['all_steps'][0])
            
            # Get and calculate metrics
            train_losses = np.array(metrics['train']['loss'])
            test_losses = np.array(metrics['test']['loss'])
            train_accs = np.array(metrics['train']['accuracy'])
            test_accs = np.array(metrics['test']['accuracy'])
            
            train_loss_mean = np.mean(train_losses, axis=0)
            test_loss_mean = np.mean(test_losses, axis=0)
            train_acc_mean = np.mean(train_accs, axis=0)
            test_acc_mean = np.mean(test_accs, axis=0)
            
            train_loss_std = np.std(train_losses, axis=0)
            test_loss_std = np.std(test_losses, axis=0)
            train_acc_std = np.std(train_accs, axis=0)
            test_acc_std = np.std(test_accs, axis=0)
            
            d_exp = np.log2(d)
            color = cmap(norm(d_exp))
            
            # Plot losses
            ax1.plot(all_steps, train_loss_mean, '--', color=color, 
                    label=f'd = 2^{int(d_exp)} (train)', lw=2)
            ax1.plot(all_steps, test_loss_mean, '-', color=color,
                    label=f'd = 2^{int(d_exp)} (val)', lw=2)
            ax1.fill_between(all_steps, train_loss_mean-train_loss_std, 
                           train_loss_mean+train_loss_std, color=color, alpha=0.075)
            ax1.fill_between(all_steps, test_loss_mean-test_loss_std,
                           test_loss_mean+test_loss_std, color=color, alpha=0.075)
            
            # Plot accuracies
            ax2.plot(all_steps, train_acc_mean, '--', color=color,
                    label=f'd = 2^{int(d_exp)} (train)', lw=2)
            ax2.plot(all_steps, test_acc_mean, '-', color=color,
                    label=f'd = 2^{int(d_exp)} (val)', lw=2)
            ax2.fill_between(all_steps, train_acc_mean-train_acc_std,
                           train_acc_mean+train_acc_std, color=color, alpha=0.075)
            ax2.fill_between(all_steps, test_acc_mean-test_acc_std,
                           test_acc_mean+test_acc_std, color=color, alpha=0.075)
        
        # Adjust scales and labels
        ax1.set_yscale('log')
        ax1.set_xlabel('Training Steps (t)', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax2.set_xlabel('Training Steps (t)', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_ylim([0, 1.02])
        ax1.set_xlim([0, 10000])
        ax2.set_xlim([0, 10000])
        
        # Add colorbars with integer ticks
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax, label='Embedding Size (log2)', 
                            ticks=range(int(min(np.log2(embedding_sizes))), int(max(np.log2(embedding_sizes)))+1))
        
        # Add legend for line styles only
        lines1 = [Line2D([0], [0], color='gray', linestyle='-', label='Validation'),
                 Line2D([0], [0], color='gray', linestyle='--', label='Train')]
        fig.legend(handles=lines1, loc='center right', bbox_to_anchor=(1.03, 0.5))

        ax1.grid(True, alpha=0.2)
        ax2.grid(True, alpha=0.2)
        plt.suptitle(f'Metrics vs Steps for {model_name.upper()} (L={layer})', 
                 fontsize=12)
        
        plt.tight_layout()
        plt.savefig(results_dir / f'{model_name.upper()}_Q5_L{layer}_combined.png', 
                   dpi=300, bbox_inches='tight', pad_inches=0.2)
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
    norm = plt.Normalize(1, 3)
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Create figure with extra space on right for colorbar and legend
    fig = plt.figure(figsize=(20, 12))  # Increased width to accommodate colorbar and legend
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot Loss vs. d (top left)
    for layer in [1, 2, 3]:
        mask = [l == layer for l in layer_values]
        ax1.plot(np.log2(np.array(d_values)[mask]), np.array(L_train_values)[mask], '-o',
                color=cmap(norm(layer)), alpha=0.7)
        ax1.plot(np.log2(np.array(d_values)[mask]), np.array(L_eval_values)[mask], '--s',
                color=cmap(norm(layer)), alpha=0.7)
    ax1.set_xlabel('Embedding dimension (log2)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. embedding dimension')
    ax1.set_yscale('log')
    ax1.set_xticks(range(int(min(np.log2(d_values))), int(max(np.log2(d_values)))+1))
    ax1.grid(True)

    # Plot Accuracy vs. d (top right)
    for layer in [1, 2, 3]:
        mask = [l == layer for l in layer_values]
        ax2.plot(np.log2(np.array(d_values)[mask]), np.array(A_train_values)[mask], '-o',
                color=cmap(norm(layer)), alpha=0.7)
        ax2.plot(np.log2(np.array(d_values)[mask]), np.array(A_eval_values)[mask], '--s',
                color=cmap(norm(layer)), alpha=0.7)
    ax2.set_xlabel('Embedding dimension (log2)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs. embedding dimension')
    ax2.set_xticks(range(int(min(np.log2(d_values))), int(max(np.log2(d_values)))+1))
    ax2.grid(True)

    # Plot tf(L) vs. d (bottom left)
    for layer in [1, 2, 3]:
        mask = [l == layer for l in layer_values]
        ax3.plot(np.log2(np.array(d_values)[mask]), np.array(tf_L_train_values)[mask], '-o',
                color=cmap(norm(layer)), alpha=0.7)
        ax3.plot(np.log2(np.array(d_values)[mask]), np.array(tf_L_eval_values)[mask], '--s',
                color=cmap(norm(layer)), alpha=0.7)
    ax3.set_xlabel('Embedding dimension (log2)')
    ax3.set_ylabel('tf(L)')
    ax3.set_title('tf(L) vs. embedding dimension')
    ax3.set_xticks(range(int(min(np.log2(d_values))), int(max(np.log2(d_values)))+1))
    ax3.grid(True)

    # Plot tf(A) vs. d (bottom right)
    for layer in [1, 2, 3]:
        mask = [l == layer for l in layer_values]
        ax4.plot(np.log2(np.array(d_values)[mask]), np.array(tf_A_train_values)[mask], '-o',
                color=cmap(norm(layer)), alpha=0.7)
        ax4.plot(np.log2(np.array(d_values)[mask]), np.array(tf_A_eval_values)[mask], '--s',
                color=cmap(norm(layer)), alpha=0.7)
    ax4.set_xlabel('Embedding dimension (log2)')
    ax4.set_ylabel('tf(A)')
    ax4.set_title('tf(A) vs. embedding dimension')
    ax4.set_xticks(range(int(min(np.log2(d_values))), int(max(np.log2(d_values)))+1))
    ax4.grid(True)

    # Add colorbar in dedicated space on right
    cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Layer', ticks=[1, 2, 3])

    # Add legend in dedicated space on right
    lines = [Line2D([0], [0], color='gray', linestyle='-', marker='o', label='Train'),
             Line2D([0], [0], color='gray', linestyle='--', marker='s', label='Validation')]
    fig.legend(handles=lines, fontsize=10, bbox_to_anchor=(1.03, 0.5), loc='upper right')

    plt.suptitle(f'Training Metrics vs. d for {model_name.upper()}', fontsize=14, y=0.95)
    plt.savefig(results_dir / f'{model_name.upper()}_metrics_vs_d.png', dpi=300, bbox_inches='tight')
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
    norm = plt.Normalize(1, 3)
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    # Create figure with extra space for colorbar and legend
    fig = plt.figure(figsize=(20, 6))  # Increased width to accommodate colorbar and legend
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Plot Training and Validation Loss vs. P (left)
    for layer in [1, 2, 3]:
        mask = [l == layer for l in layer_values]
        ax1.plot(np.array(P_values)[mask], np.array(L_train_values)[mask], '-o',
                color=cmap(norm(layer)), alpha=0.7)
        ax1.plot(np.array(P_values)[mask], np.array(L_eval_values)[mask], '--s',
                color=cmap(norm(layer)), alpha=0.7)
    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. Parameters')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)
    
    # Plot Training and Validation Accuracy vs. P (right)
    for layer in [1, 2, 3]:
        mask = [l == layer for l in layer_values]
        ax2.plot(np.array(P_values)[mask], np.array(A_train_values)[mask], '-o',
                color=cmap(norm(layer)), alpha=0.7)
        ax2.plot(np.array(P_values)[mask], np.array(A_eval_values)[mask], '--s',
                color=cmap(norm(layer)), alpha=0.7)
    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs. Parameters')
    ax2.set_xscale('log')
    ax2.grid(True)
    
    # Add colorbar in dedicated space on right
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.755])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Layer', ticks=[1, 2, 3])
    
    # Add legend in dedicated space on right
    lines = [Line2D([0], [0], color='gray', linestyle='-', marker='o', label='Train'),
             Line2D([0], [0], color='gray', linestyle='--', marker='s', label='Validation')]
    fig.legend(handles=lines, fontsize=10, bbox_to_anchor=(1.03, 0.5), loc='upper right')
    
    plt.suptitle(f'Training Metrics vs. Parameters for {model_name.upper()}', fontsize=14, y=0.95)
    plt.savefig(results_dir / f'{model_name.upper()}_metrics_vs_params.png', dpi=300, bbox_inches='tight')
    plt.close()
        
def plot_loss_accuracy_q6_a(metrics_per_batch_size, model_name, results_dir):
    """Plot training and validation curves for Q6, with all batch sizes on the same figures."""
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create color map for different batch sizes
    batch_sizes = sorted(metrics_per_batch_size.keys())
    alphas = sorted(metrics_per_batch_size[batch_sizes[0]].keys())
    print(f"==== alphas: \n{alphas} =====")
    colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))

    # Convert alphas to alpha*T values
    T = 10000
    alpha_T_values = [alpha * T for alpha in alphas]

    # Create figure with extra space for colorbar and legend
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    for batch_size in batch_sizes:
        # Get metrics
        train_losses = [metrics_per_batch_size[batch_size][alpha]['min_train_loss'] for alpha in alphas]
        test_losses = [metrics_per_batch_size[batch_size][alpha]['min_test_loss'] for alpha in alphas]
        train_accs = [metrics_per_batch_size[batch_size][alpha]['max_train_accuracy'] for alpha in alphas]
        test_accs = [metrics_per_batch_size[batch_size][alpha]['max_test_accuracy'] for alpha in alphas]
        
        # Get color for this batch size
        color_index = batch_sizes.index(batch_size)
        color = colors[color_index]
        
        # Plot losses
        ax1.plot(alpha_T_values, train_losses, '--', color=color, label=f'batch={batch_size} (train)', lw=2.0)
        ax1.plot(alpha_T_values, test_losses, '-', color=color, label=f'batch={batch_size} (val)', lw=2.0)
        
        # Plot accuracies  
        ax2.plot(alpha_T_values, train_accs, '--', color=color, label=f'batch={batch_size} (train)', lw=2.0)
        ax2.plot(alpha_T_values, test_accs, '-', color=color, label=f'batch={batch_size} (val)', lw=2.0)

    # Configure loss axis
    ax1.set_yscale('log')
    ax1.set_xlabel('αT', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs αT', fontsize=12)
    ax1.grid(True)
    ax1.set_xlim([1000, 10000])

    # Configure accuracy axis
    ax2.set_ylim([0.0, 1.02])
    ax2.set_xlabel('αT', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12) 
    ax2.set_title('Accuracy vs αT', fontsize=12)
    ax2.grid(True)
    ax2.set_xlim([1000, 10000])

    # Add colorbar
    norm = plt.Normalize(min(np.log2(batch_sizes)), max(np.log2(batch_sizes)))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.755])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Batch Size (log2)', 
                       ticks=range(int(min(np.log2(batch_sizes))), int(max(np.log2(batch_sizes)))+1))

    # Add legend
    lines = [Line2D([0], [0], color='gray', linestyle='-', label='Validation'),
             Line2D([0], [0], color='gray', linestyle='--', label='Train')]
    fig.legend(handles=lines, fontsize=10, bbox_to_anchor=(1.05, 0.5), loc='center right')

    plt.suptitle(f'Training Metrics vs αT for {model_name.upper()}', fontsize=14, y=0.95)
    plt.savefig(results_dir / f'{model_name.upper()}_Q6_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
        
def plot_loss_accuracy_q6_b(metrics_per_batch_size, model_name, results_dir):
    """Plot training and validation curves for Q6, with batch sizes on the x-axis and different colors for each alpha."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with extra space for colorbar and legend
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # Loss plot
    ax2 = fig.add_subplot(gs[0, 1])  # Accuracy plot
    
    batch_sizes = sorted(metrics_per_batch_size.keys())
    alphas = sorted(metrics_per_batch_size[batch_sizes[0]].keys())
    batch_size_values = [int(bs) for bs in batch_sizes]
    
    # Create colormap for alpha values
    norm = plt.Normalize(min(alphas), max(alphas))
    cmap = plt.cm.viridis
    
    for alpha in alphas:
        color = cmap(norm(alpha))
        
        # Get metrics
        train_losses = [metrics_per_batch_size[bs][alpha]['min_train_loss'] for bs in batch_sizes]
        val_losses = [metrics_per_batch_size[bs][alpha]['min_test_loss'] for bs in batch_sizes]
        train_accs = [metrics_per_batch_size[bs][alpha]['max_train_accuracy'] for bs in batch_sizes]
        val_accs = [metrics_per_batch_size[bs][alpha]['max_test_accuracy'] for bs in batch_sizes]
        
        # Plot losses
        ax1.plot(batch_size_values, train_losses, 'o-', color=color, label=f'α={alpha:.1f}', lw=2)
        ax1.plot(batch_size_values, val_losses, 's-', color=color, lw=2)
        
        # Plot accuracies
        ax2.plot(batch_size_values, train_accs, 'o-', color=color, label=f'α={alpha:.1f}', lw=2)
        ax2.plot(batch_size_values, val_accs, 's-', color=color, lw=2)
    
    # Configure loss axis
    ax1.set_yscale('log')
    ax1.set_xlabel('Batch Size (log2)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs Batch Size', fontsize=12)
    ax1.grid(True)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([2**i for i in range(5,10)])
    ax1.set_xticklabels([f'{i}' for i in range(5,10)])
    ax1.set_xlim(2**5, 2**9)
    
    # Configure accuracy axis  
    ax2.set_ylim([0.0, 1.02])
    ax2.set_xlabel('Batch Size (log2)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Batch Size', fontsize=12)
    ax2.grid(True)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([2**i for i in range(5,10)])
    ax2.set_xticklabels([f'{i}' for i in range(5,10)])
    ax2.set_xlim(2**5, 2**9)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='α', ticks=np.arange(0.1, 1.1, 0.1))
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', label='Train', markersize=8),
        Line2D([0], [0], marker='s', color='gray', label='Validation', markersize=8)
    ]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.05, 0.5))
    
    plt.suptitle(f'Metrics vs Batch Size for {model_name.upper()}', fontsize=14)
    plt.savefig(results_dir / f'{model_name.upper()}_Q6_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_accuracy_q7_a(metrics_per_weight_decay, results_dir, model_name="lstm"):
    """Plot training and validation curves for Q7, with all weight decays on the same figures."""
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create color map for different weight decays
    weight_decays = sorted(metrics_per_weight_decay.keys())
    norm = plt.Normalize(0.25, 1.0)
    cmap = plt.cm.viridis
    colors = [cmap(norm(wd)) for wd in weight_decays]

    # Create figure with extra space for colorbar and legend
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.2)
    ax1 = fig.add_subplot(gs[0])  # L2 norm
    ax2 = fig.add_subplot(gs[1])  # Loss
    ax3 = fig.add_subplot(gs[2])  # Accuracy

    for weight_decay, color in zip(weight_decays, colors):
        all_steps = metrics_per_weight_decay[weight_decay]['all_steps'][0]
        
        # Plot L2 norm
        train_l2_norms = np.array(metrics_per_weight_decay[weight_decay]['train']['l2_norm'])
        train_mean = np.mean(train_l2_norms, axis=0)
        train_std = np.std(train_l2_norms, axis=0)
        ax1.plot(all_steps, train_mean, '-', color=color, lw=2.0)
        ax1.fill_between(all_steps, train_mean-train_std, train_mean+train_std, color=color, alpha=0.2)
        
        # Plot losses
        train_losses = np.array(metrics_per_weight_decay[weight_decay]['train']['loss'])
        val_losses = np.array(metrics_per_weight_decay[weight_decay]['test']['loss'])
        train_mean = np.mean(train_losses, axis=0)
        val_mean = np.mean(val_losses, axis=0)
        train_std = np.std(train_losses, axis=0)
        val_std = np.std(val_losses, axis=0)
        
        ax2.plot(all_steps, train_mean, '--', color=color, lw=2.0)
        ax2.plot(all_steps, val_mean, '-', color=color, lw=2.0)
        ax2.fill_between(all_steps, train_mean-train_std, train_mean+train_std, color=color, alpha=0.2)
        ax2.fill_between(all_steps, val_mean-val_std, val_mean+val_std, color=color, alpha=0.2)
        
        # Plot accuracies
        train_accs = np.array(metrics_per_weight_decay[weight_decay]['train']['accuracy'])
        val_accs = np.array(metrics_per_weight_decay[weight_decay]['test']['accuracy'])
        train_mean = np.mean(train_accs, axis=0)
        val_mean = np.mean(val_accs, axis=0)
        train_std = np.std(train_accs, axis=0)
        val_std = np.std(val_accs, axis=0)
        
        ax3.plot(all_steps, train_mean, '--', color=color, lw=2.0)
        ax3.plot(all_steps, val_mean, '-', color=color, lw=2.0)
        ax3.fill_between(all_steps, train_mean-train_std, train_mean+train_std, color=color, alpha=0.2)
        ax3.fill_between(all_steps, val_mean-val_std, val_mean+val_std, color=color, alpha=0.2)

    # Configure axes
    for ax in [ax1, ax2]:
        ax.set_yscale('log')
    ax3.set_ylim([0.0, 1.02])
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 10000)
        ax.set_xlabel('Training Steps (t)', fontsize=12)
        ax.grid(True)

    ax1.set_ylabel('L2 Norm', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12) 
    ax3.set_ylabel('Accuracy', fontsize=12)

    ax1.set_title('L2 Norm vs Steps', fontsize=12)
    ax2.set_title('Loss vs Steps', fontsize=12)
    ax3.set_title('Accuracy vs Steps', fontsize=12)

    # Add colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Weight Decay', ticks=np.arange(0.25, 1.25, 0.25))

    # Add legend
    legend_elements = [
        Line2D([0], [0], linestyle='--', color='gray', label='Train'),
        Line2D([0], [0], linestyle='-', color='gray', label='Validation')
    ]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.05, 0.5))

    plt.suptitle(f'Metrics vs Steps for {model_name.upper()}', fontsize=14)
    plt.savefig(results_dir / f'{model_name.upper()}_Q7_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_accuracy_q7_b(metrics_per_weight_decay, results_dir, model_name="lstm"):
    """Plot loss, accuracy, and tf curves for Q7, as a function of weight decay. Train and eval on the same figure."""
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    weight_decays = sorted(metrics_per_weight_decay.keys())

    # Extract values for each metric
    train_losses = [metrics_per_weight_decay[wd]['min_train_loss'] for wd in weight_decays]
    val_losses = [metrics_per_weight_decay[wd]['min_test_loss'] for wd in weight_decays]
    train_accs = [metrics_per_weight_decay[wd]['max_train_accuracy'] for wd in weight_decays]
    val_accs = [metrics_per_weight_decay[wd]['max_test_accuracy'] for wd in weight_decays]
    tf_losses = [metrics_per_weight_decay[wd]['min_train_loss_step'] for wd in weight_decays]
    tf_accs = [metrics_per_weight_decay[wd]['max_train_accuracy_step'] for wd in weight_decays]
    tf_losses_val = [metrics_per_weight_decay[wd]['min_test_loss_step'] for wd in weight_decays]
    tf_accs_val = [metrics_per_weight_decay[wd]['max_test_accuracy_step'] for wd in weight_decays]
    
    # Create figure with 2x2 subplots and extra space for legend
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.25, wspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])  # Loss plot
    ax2 = fig.add_subplot(gs[0, 1])  # Accuracy plot
    ax3 = fig.add_subplot(gs[1, 0])  # tf(L) plot
    ax4 = fig.add_subplot(gs[1, 1])  # tf(A) plot
    
    # Plot Loss
    ax1.plot(weight_decays, train_losses, '--', label='Train', lw=2.0)
    ax1.plot(weight_decays, val_losses, '-', label='Validation', lw=2.0)
    ax1.set_yscale('log')
    ax1.set_xlabel('Weight Decay', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs Weight Decay', fontsize=12)
    ax1.grid(True)
    ax1.set_xlim(0.25, 1.0)
    ax1.set_xticks([0.25, 0.5, 0.75, 1.0])

    # Plot Accuracy
    ax2.plot(weight_decays, train_accs, '--', label='Train', lw=2.0)
    ax2.plot(weight_decays, val_accs, '-', label='Validation', lw=2.0)
    ax2.set_xlabel('Weight Decay', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Weight Decay', fontsize=12)
    ax2.grid(True)
    ax2.set_xlim(0.25, 1.0)
    ax2.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax2.set_ylim(0, 1.02)

    # Plot tf(L)
    ax3.plot(weight_decays, tf_losses, '--', label='Train', lw=2.0)
    ax3.plot(weight_decays, tf_losses_val, '-', label='Validation', lw=2.0)
    ax3.set_xlabel('Weight Decay', fontsize=12)
    ax3.set_ylabel('tf(L)', fontsize=12)
    ax3.set_title('tf(L) vs Weight Decay', fontsize=12)
    ax3.grid(True)
    ax3.set_xlim(0.25, 1.0)
    ax3.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax3.set_ylim(0, 10010)

    # Plot tf(A)
    ax4.plot(weight_decays, tf_accs, '--', label='Train', lw=2.0)
    ax4.plot(weight_decays, tf_accs_val, '-', label='Validation', lw=2.0)
    ax4.set_xlabel('Weight Decay', fontsize=12)
    ax4.set_ylabel('tf(A)', fontsize=12)
    ax4.set_title('tf(A) vs Weight Decay', fontsize=12)
    ax4.grid(True)
    ax4.set_xlim(0.25, 1.0)
    ax4.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax4.set_ylim(0, 10010)
    # Add legend
    legend_elements = [
        Line2D([0], [0], linestyle='--', color='gray', label='Train'),
        Line2D([0], [0], linestyle='-', color='gray', label='Validation')
    ]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.01, 0.5), fontsize=12)

    plt.suptitle(f'Training Metrics vs Weight Decay for {model_name.upper()}', fontsize=18)
    plt.savefig(results_dir / f'{model_name.upper()}_Q7_combined.png', dpi=300, bbox_inches='tight')
    plt.close()