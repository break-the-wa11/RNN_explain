"""
Reproduce Figure 7 from "Deep Thinking" paper (arXiv:2106.04537)

This script visualizes how the maze solution evolves over RNN iterations.
It runs the RNN block for 1 to N iterations and shows the predicted path
at each iteration, demonstrating how the solution progressively improves.

The key insight is that the RNN iteratively refines the solution,
similar to how iterative optimization algorithms converge to a solution.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import OrderedDict

# ============================================================================
# Setup paths
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# 1. RUN RNN FOR VARIABLE ITERATIONS AND GET PREDICTIONS
# ============================================================================

def run_rnn_iterations(model, x_input, max_iterations, device):
    """
    Run the RNN block for 1 to max_iterations and collect:
    - Hidden states (x_out) at each iteration
    - Predicted paths (after passing through head) at each iteration
    
    Args:
        model: DTNet model
        x_input: Input maze tensor, shape (B, 3, H, W)
        max_iterations: Maximum number of RNN iterations
        device: torch.device
    
    Returns:
        predictions: List of predicted path tensors at each iteration
        hidden_states: List of hidden states (x_out) at each iteration
    """
    model.eval()
    x_input = x_input.to(device)
    
    predictions = []
    hidden_states = []
    
    with torch.no_grad():
        # Initial projection: x_input -> initial hidden state
        x_5 = torch.relu(model.projection(x_input))
        
        for t in range(max_iterations):
            # One RNN iteration
            # Step 1: Concatenate hidden state with input and apply first conv
            x_1 = model.recur_block[0](torch.cat([x_5, x_input], dim=1))
            
            # Step 2: First BasicBlock
            conv_x1 = model.recur_block[1][0].conv1(x_1)
            x_21 = torch.relu(conv_x1)
            conv_x21 = model.recur_block[1][0].conv2(x_21)
            x_3 = torch.relu(conv_x21 + x_1)  # Residual connection
            
            # Step 3: Second BasicBlock
            conv_x3 = model.recur_block[1][1].conv1(x_3)
            x_41 = torch.relu(conv_x3)
            conv_x41 = model.recur_block[1][1].conv2(x_41)
            x_5 = torch.relu(conv_x41 + x_3)  # Residual connection
            
            # Store hidden state
            hidden_states.append(x_5.clone())
            
            # Pass through head to get prediction
            # head: Conv2d(128->32) -> ReLU -> Conv2d(32->8) -> ReLU -> Conv2d(8->2)
            output = model.head(x_5)  # Shape: (B, 2, H, W)
            predictions.append(output.clone())
    
    return predictions, hidden_states


def get_predicted_path(output):
    """
    Convert network output to predicted path.
    
    Args:
        output: Network output tensor, shape (B, 2, H, W)
                Channel 0: logits for "not path"
                Channel 1: logits for "path"
    
    Returns:
        predicted_path: Binary tensor, shape (B, H, W)
                        1 where path is predicted, 0 otherwise
    """
    # Argmax over channel dimension
    # If channel 1 > channel 0, predict path (1), else not path (0)
    predicted_path = torch.argmax(output, dim=1)  # Shape: (B, H, W)
    return predicted_path


# ============================================================================
# 2. VISUALIZATION FUNCTIONS
# ============================================================================

def create_maze_with_path_overlay(maze_input, predicted_path, ground_truth=None):
    """
    Create visualization of maze with predicted path overlaid.
    
    Args:
        maze_input: Input maze, shape (H, W, 3) or (3, H, W)
        predicted_path: Predicted path, shape (H, W), binary
        ground_truth: Optional ground truth path, shape (H, W)
    
    Returns:
        overlay: RGB image with path overlay
    """
    # Ensure maze_input is (H, W, 3)
    if maze_input.shape[0] == 3:
        maze_input = maze_input.transpose(1, 2, 0)
    
    # Create overlay image
    overlay = maze_input.copy()
    
    # Add predicted path in blue
    path_mask = predicted_path > 0.5
    overlay[path_mask, 0] = 0.0  # Red channel
    overlay[path_mask, 1] = 0.0  # Green channel  
    overlay[path_mask, 2] = 1.0  # Blue channel
    
    # If ground truth provided, show it in orange where it differs
    if ground_truth is not None:
        gt_mask = ground_truth > 0.5
        # Show correct predictions in orange
        correct_mask = path_mask & gt_mask
        overlay[correct_mask, 0] = 1.0
        overlay[correct_mask, 1] = 1.0
        overlay[correct_mask, 2] = 0.0
    
    return overlay


def plot_iteration_progression(maze_input, predictions, ground_truth=None,
                                iterations_to_show=None, save_path=None,
                                title="Maze Solution Over RNN Iterations"):
    """
    Plot the predicted path at multiple iterations, similar to Figure 7.
    
    Args:
        maze_input: Input maze, numpy array (H, W, 3)
        predictions: List of prediction tensors at each iteration
        ground_truth: Optional ground truth solution
        iterations_to_show: List of iteration indices to display (0-indexed)
                           If None, shows iterations [0, 1, 2, 4, 9, 19, 29, ...] 
        save_path: Path to save the figure
        title: Figure title
    """
    num_iters = len(predictions)
    
    # Default iterations to show (similar to paper: showing progression)
    if iterations_to_show is None:
        # Show: 1, 2, 3, 5, 10, 20, 30, ... (1-indexed in display)
        iterations_to_show = [0, 1, 2, 4, 9]
        if num_iters > 20:
            iterations_to_show.append(19)
        if num_iters > 30:
            iterations_to_show.append(29)
        if num_iters > 50:
            iterations_to_show.append(49)
        if num_iters > 100:
            iterations_to_show.append(99)
        # Filter to valid indices
        iterations_to_show = [i for i in iterations_to_show if i < num_iters]
    
    n_plots = len(iterations_to_show) + 2  # +2 for input and ground truth
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3))
    
    # Plot 1: Input maze
    axes[0].imshow(maze_input)
    axes[0].set_title('Input Maze', fontsize=10)
    axes[0].axis('off')
    
    # Plot predictions at each iteration
    for idx, iter_num in enumerate(iterations_to_show):
        pred = predictions[iter_num]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu()
        
        # Get predicted path (argmax)
        pred_path = get_predicted_path(pred)
        pred_path_np = pred_path[0].numpy()  # Remove batch dimension
        
        # Create overlay
        overlay = create_maze_with_path_overlay(maze_input, pred_path_np)
        
        axes[idx + 1].imshow(overlay)
        axes[idx + 1].set_title(f'Iter {iter_num + 1}', fontsize=10)
        axes[idx + 1].axis('off')
    
    # Plot ground truth (if available)
    if ground_truth is not None:
        overlay_gt = create_maze_with_path_overlay(maze_input, ground_truth)
        # Show ground truth in orange
        gt_mask = ground_truth > 0.5
        overlay_gt[gt_mask, 0] = 1.0
        overlay_gt[gt_mask, 1] = 1.0
        overlay_gt[gt_mask, 2] = 0.0
        axes[-1].imshow(overlay_gt)
        axes[-1].set_title('Ground Truth', fontsize=10)
    else:
        axes[-1].imshow(maze_input)
        axes[-1].set_title('Input', fontsize=10)
    axes[-1].axis('off')
    
    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_iteration_grid(maze_input, predictions, ground_truth=None,
                        max_iters_to_show=30, save_path=None,
                        title="Maze Solution Evolution"):
    """
    Plot predictions in a grid format showing all iterations up to max_iters_to_show.
    
    Args:
        maze_input: Input maze, numpy array (H, W, 3)
        predictions: List of prediction tensors
        ground_truth: Optional ground truth
        max_iters_to_show: Maximum number of iterations to display
        save_path: Path to save figure
        title: Figure title
    """
    n_iters = min(len(predictions), max_iters_to_show)
    
    # Calculate grid dimensions
    n_cols = min(10, n_iters + 2)  # +2 for input and GT
    n_rows = (n_iters + 2 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # Plot input
    axes[0].imshow(maze_input)
    axes[0].set_title('Input', fontsize=8)
    axes[0].axis('off')
    
    # Plot each iteration
    for i in range(n_iters):
        pred = predictions[i]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu()
        
        pred_path = get_predicted_path(pred)
        pred_path_np = pred_path[0].numpy()
        
        overlay = create_maze_with_path_overlay(maze_input, pred_path_np)
        
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f't={i + 1}', fontsize=8)
        axes[i + 1].axis('off')
    
    # Plot ground truth
    if ground_truth is not None:
        gt_overlay = maze_input.copy()
        gt_mask = ground_truth > 0.5
        gt_overlay[gt_mask, 0] = 1.0
        gt_overlay[gt_mask, 1] = 1.0
        gt_overlay[gt_mask, 2] = 0.0
        axes[n_iters + 1].imshow(gt_overlay)
        axes[n_iters + 1].set_title('GT', fontsize=8)
        axes[n_iters + 1].axis('off')
    
    # Hide unused subplots
    for i in range(n_iters + 2, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_accuracy_vs_iteration(predictions, ground_truth, save_path=None,
                                title="Path Prediction Accuracy vs Iteration"):
    """
    Plot accuracy of path prediction at each iteration.
    
    Args:
        predictions: List of prediction tensors
        ground_truth: Ground truth path, shape (H, W)
        save_path: Path to save figure
        title: Figure title
    """
    accuracies = []
    path_recalls = []
    path_precisions = []
    
    gt_tensor = torch.tensor(ground_truth).float()
    gt_path_pixels = (gt_tensor > 0.5).sum().item()
    
    for pred in predictions:
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu()
        
        pred_path = get_predicted_path(pred)[0].float()  # (H, W)
        
        # Overall accuracy
        correct = (pred_path == gt_tensor).sum().item()
        total = gt_tensor.numel()
        accuracies.append(correct / total * 100)
        
        # Path recall (what fraction of true path is predicted)
        true_positives = ((pred_path > 0.5) & (gt_tensor > 0.5)).sum().item()
        if gt_path_pixels > 0:
            path_recalls.append(true_positives / gt_path_pixels * 100)
        else:
            path_recalls.append(0)
        
        # Path precision (what fraction of predicted path is correct)
        pred_path_pixels = (pred_path > 0.5).sum().item()
        if pred_path_pixels > 0:
            path_precisions.append(true_positives / pred_path_pixels * 100)
        else:
            path_precisions.append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(1, len(predictions) + 1)
    
    ax.plot(iterations, accuracies, 'b-', linewidth=2, label='Overall Accuracy')
    ax.plot(iterations, path_recalls, 'g--', linewidth=2, label='Path Recall')
    ax.plot(iterations, path_precisions, 'r:', linewidth=2, label='Path Precision')
    
    ax.set_xlabel('RNN Iteration', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(predictions))
    ax.set_ylim(0, 105)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig, {'accuracies': accuracies, 'recalls': path_recalls, 'precisions': path_precisions}


def plot_figure7_style(maze_input, predictions, ground_truth=None,
                       iterations_to_show=[0, 1, 2, 4, 9, 19, 29],
                       save_path=None):
    """
    Create a Figure 7 style visualization from the Deep Thinking paper.
    
    Shows the maze with predicted path overlaid at selected iterations,
    demonstrating how the solution progressively improves.
    """
    # Filter valid iterations
    iterations_to_show = [i for i in iterations_to_show if i < len(predictions)]
    
    n_cols = len(iterations_to_show) + 1  # +1 for ground truth
    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 2.5))
    
    # Custom colormap: black for walls, white for empty, red for predicted path
    for idx, iter_num in enumerate(iterations_to_show):
        pred = predictions[iter_num]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu()
        
        pred_path = get_predicted_path(pred)
        pred_path_np = pred_path[0].numpy()
        
        # Create visualization
        # Start with the maze
        vis = maze_input.copy()
        
        # Overlay predicted path in blue
        path_mask = pred_path_np > 0.5
        vis[path_mask] = [0.0, 0.0, 1.0]  # Blue
        
        axes[idx].imshow(vis)
        axes[idx].set_title(f'Iteration {iter_num + 1}', fontsize=10)
        axes[idx].axis('off')
    
    # Ground truth in last panel
    if ground_truth is not None:
        vis_gt = maze_input.copy()
        gt_mask = ground_truth > 0.5
        vis_gt[gt_mask] = [1.0, 1.0, 0.0]  # Orange
        axes[-1].imshow(vis_gt)
        axes[-1].set_title('Ground Truth', fontsize=10)
    else:
        axes[-1].imshow(maze_input)
        axes[-1].set_title('Input', fontsize=10)
    axes[-1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ============================================================================
# 3. MAIN FUNCTION
# ============================================================================

def visualize_iteration_progression(model, x_input, ground_truth=None,
                                     max_iterations=30, device='cuda',
                                     sample_idx=0, save_prefix=None):
    """
    Complete pipeline to visualize maze solution over RNN iterations.
    
    Args:
        model: Trained DTNet model
        x_input: Input tensor (B, C, H, W) or (B, H, W, C)
        ground_truth: Ground truth solution (H, W) or None
        max_iterations: Number of RNN iterations to run
        device: Computing device
        sample_idx: Which sample in batch to visualize
        save_prefix: Prefix for saved files
    
    Returns:
        Dictionary with predictions, hidden states, and metrics
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.eval()
    
    # Handle input format
    if isinstance(x_input, np.ndarray):
        x_input = torch.tensor(x_input, dtype=torch.float32)
    
    # Ensure (B, C, H, W) format
    if x_input.dim() == 3:
        x_input = x_input.unsqueeze(0)
    if x_input.shape[-1] == 3:  # (B, H, W, C) -> (B, C, H, W)
        x_input = x_input.permute(0, 3, 1, 2)
    
    x_input = x_input.to(device)
    
    # Get input as numpy for visualization
    maze_input_np = x_input[sample_idx].cpu().numpy().transpose(1, 2, 0)
    
    print("="*60)
    print("Maze Solution Iteration Visualization")
    print("="*60)
    print(f"Running {max_iterations} RNN iterations...")
    
    # Run RNN
    predictions, hidden_states = run_rnn_iterations(model, x_input, max_iterations, device)
    print(f"Collected {len(predictions)} predictions")
    
    # Set up save paths
    if save_prefix is None:
        save_prefix = os.path.join(RESULTS_DIR, 'maze_iterations')
    
    # Plot 1: Figure 7 style - key iterations
    print("\nGenerating Figure 7 style visualization...")
    plot_figure7_style(
        maze_input_np, predictions, ground_truth,
        iterations_to_show=[0, 1, 2, 4, 9, 19, 29],
        save_path=f'{save_prefix}_fig7_style.png'
    )
    
    # Plot 2: Progression with more iterations
    print("\nGenerating progression plot...")
    plot_iteration_progression(
        maze_input_np, predictions, ground_truth,
        iterations_to_show=[0, 1, 2, 4, 9, 14, 19, 29],
        save_path=f'{save_prefix}_progression.png',
        title="Maze Solution Over RNN Iterations"
    )
    
    # Plot 3: Grid of all iterations
    print("\nGenerating iteration grid...")
    plot_iteration_grid(
        maze_input_np, predictions, ground_truth,
        max_iters_to_show=30,
        save_path=f'{save_prefix}_grid.png',
        title="Maze Solution Evolution (All Iterations)"
    )
    
    # Plot 4: Accuracy over iterations (if ground truth available)
    metrics = None
    if ground_truth is not None:
        print("\nComputing accuracy metrics...")
        _, metrics = plot_accuracy_vs_iteration(
            predictions, ground_truth,
            save_path=f'{save_prefix}_accuracy.png',
            title="Path Prediction Accuracy vs Iteration"
        )
    
    print(f"\nDone! Results saved with prefix: {save_prefix}")
    
    return {
        'predictions': predictions,
        'hidden_states': hidden_states,
        'maze_input': maze_input_np,
        'ground_truth': ground_truth,
        'metrics': metrics
    }


# ============================================================================
# 4. MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
    
    from deepthinking.models import dt_net_recall_2d
    
    # Paths
    PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'outputs/mazes_ablation/training-rusty-Tayla/model_best.pth')
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data/maze_data_test_9')
    
    # Parameters
    MAX_ITERATIONS = 30
    EXAMPLE_IDX = 0  # Which maze to visualize
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    print("\nLoading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    model = dt_net_recall_2d(width=128, in_channels=3, max_iters=MAX_ITERATIONS)
    
    state_dict = checkpoint['net']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\nLoading maze data...")
    x_np = np.load(os.path.join(DATA_PATH, 'inputs.npy'))
    y_np = np.load(os.path.join(DATA_PATH, 'solutions.npy'))
    
    print(f"Data shapes: inputs={x_np.shape}, solutions={y_np.shape}")
    
    # Select example
    maze_input = x_np[EXAMPLE_IDX]  # Shape: (H, W, 3)
    ground_truth = y_np[EXAMPLE_IDX]  # Shape: (H, W) - binary path
    
    # Handle ground truth format (might be (H, W, 1) or (H, W))
    if ground_truth.ndim == 3:
        ground_truth = ground_truth[:, :, 0]
    
    print(f"Selected maze {EXAMPLE_IDX}: input shape={maze_input.shape}, gt shape={ground_truth.shape}")
    
    # -------------------------------------------------------------------------
    # Visualize
    # -------------------------------------------------------------------------
    x_input = torch.tensor(maze_input, dtype=torch.float32).unsqueeze(0)  # (1, H, W, 3)
    
    results = visualize_iteration_progression(
        model=model,
        x_input=x_input,
        ground_truth=ground_truth,
        max_iterations=MAX_ITERATIONS,
        device=device,
        sample_idx=0,
        save_prefix=os.path.join(RESULTS_DIR, f'maze_{EXAMPLE_IDX}_iterations')
    )
    
    # -------------------------------------------------------------------------
    # Visualize multiple mazes
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Visualizing multiple maze examples...")
    print("="*60)
    
    for idx in [0, 1, 2]:
        if idx >= len(x_np):
            break
            
        maze_input = x_np[idx]
        ground_truth = y_np[idx]
        if ground_truth.ndim == 3:
            ground_truth = ground_truth[:, :, 0]
        
        x_input = torch.tensor(maze_input, dtype=torch.float32).unsqueeze(0)
        
        print(f"\n--- Maze {idx} ---")
        results = visualize_iteration_progression(
            model=model,
            x_input=x_input,
            ground_truth=ground_truth,
            max_iterations=MAX_ITERATIONS,
            device=device,
            sample_idx=0,
            save_prefix=os.path.join(RESULTS_DIR, f'maze_{idx}_iterations')
        )
    
    print("\nAll visualizations complete!")