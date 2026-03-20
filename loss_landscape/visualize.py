"""
Loss Landscape Visualization with RNN Trajectory
Visualizes J_alpha loss and shows how RNN iterations descend on the landscape.

File location: loss_landscape/visualize.py
Output location: loss_landscape/results/
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy
from tqdm import tqdm

# ============================================================================
# Setup paths - this file should be in loss_landscape/ folder
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Results will be saved in loss_landscape/results/
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# 1. LOSS FUNCTION J_alpha
# ============================================================================

def compute_J_alpha(model, x_input, h1, h2, h3, h4, x_out, alpha, device):
    """
    Compute J_alpha loss given the hidden states and model weights.

    Returns a tensor of shape (batch_size,) - one loss value per sample.
    All computations are done on the specified device (GPU).
    
    J_alpha = ||h1 - W0*x_out - b||^2 + ||h2 - W1*h1||^2 + ||h3 - W2*h2 - h1||^2 
              + ||h4 - W3*h3||^2 + ||x_out - W4*h4 - h3||^2
              - (||[h2 - W1*h1 - alpha*h2]_+||^2 + ||[h3 - W2*h2 - h1 - alpha*h3]_+||^2 
                 + ||[h4 - W3*h3 - alpha*h4]_+||^2 + ||[x_out - W4*h4 - h3 - alpha*x_out]_+||^2)
    
    Mapping to the network:
    - h1 = x_1 (output of recur_block[0])
    - h2 = x_21 (after first ReLU)
    - h3 = x_3 (after residual connection)
    - h4 = x_41 (after second BasicBlock conv1)
    - x_out = x_5 (final output)

    Args:
        model: The trained RNN model (should be on device)
        x_input: Input tensor (B, C_in, H, W) on device
        h1, h2, h3, h4, x_out: Hidden states, each (B, C, H, W) on device
        alpha: The alpha parameter
        device: torch.device for computation
    
    Returns:
        J_alpha: Tensor of shape (batch_size,) - loss for each sample
    """
    # Ensure all tensors are on the correct device
    x_input = x_input.to(device)
    h1 = h1.to(device)
    h2 = h2.to(device)
    h3 = h3.to(device)
    h4 = h4.to(device)
    x_out = x_out.to(device)
    
    with torch.no_grad():
        # Compute intermediate convolutions based on provided hidden states
        # W0: recur_block[0] - maps concatenated [x_out_prev, x_input] to h1
        # For landscape, we use provided h1 directly
        
        # W1*h1: conv1 of BasicBlock[0]
        conv_h1 = model.recur_block[1][0].conv1(h1)  # W1 * h1
        
        # W2*h2: conv2 of BasicBlock[0]
        conv_h2 = model.recur_block[1][0].conv2(h2)  # W2 * h2
        
        # W3*h3: conv1 of BasicBlock[1]
        conv_h3 = model.recur_block[1][1].conv1(h3)  # W3 * h3
        
        # W4*h4: conv2 of BasicBlock[1]
        conv_h4 = model.recur_block[1][1].conv2(h4)  # W4 * h4
        
        # Term 1: ||h1 - W0*x_out - b||^2
        # Here W0 operation includes concatenation with input
        W0_x = model.recur_block[0](torch.cat([x_out, x_input], dim=1))
        term1 = torch.norm(h1 - W0_x, p=2, dim=(1, 2, 3))**2
        
        # Term 2: ||h2 - W1*h1||^2
        term2 = torch.norm(h2 - conv_h1, p=2, dim=(1, 2, 3))**2
        
        # Term 3: ||h3 - W2*h2 - h1||^2
        term3 = torch.norm(h3 - conv_h2 - h1, p=2, dim=(1, 2, 3))**2
        
        # Term 4: ||h4 - W3*h3||^2
        term4 = torch.norm(h4 - conv_h3, p=2, dim=(1, 2, 3))**2
        
        # Term 5: ||x_out - W4*h4 - h3||^2
        term5 = torch.norm(x_out - conv_h4 - h3, p=2, dim=(1, 2, 3))**2
        
        # Positive part terms (ReLU clipping terms)
        # [h2 - W1*h1 - alpha*h2]_+ = [h2(1-alpha) - W1*h1]_+
        relu_term2 = torch.norm(torch.relu(h2 - conv_h1 - alpha * h2), p=2, dim=(1, 2, 3))**2
        
        # [h3 - W2*h2 - h1 - alpha*h3]_+
        relu_term3 = torch.norm(torch.relu(h3 - conv_h2 - h1 - alpha * h3), p=2, dim=(1, 2, 3))**2
        
        # [h4 - W3*h3 - alpha*h4]_+
        relu_term4 = torch.norm(torch.relu(h4 - conv_h3 - alpha * h4), p=2, dim=(1, 2, 3))**2
        
        # [x_out - W4*h4 - h3 - alpha*x_out]_+
        relu_term5 = torch.norm(torch.relu(x_out - conv_h4 - h3 - alpha * x_out), p=2, dim=(1, 2, 3))**2
        
        # Full J_alpha - shape (batch_size,)
        J_alpha = (term1 + term2 + term3 + term4 + term5 
                   - (relu_term2 + relu_term3 + relu_term4 + relu_term5))
        
    return J_alpha

# ============================================================================
# 2. COLLECT RNN TRAJECTORY (All iterations)
# ============================================================================

def collect_rnn_trajectory(model, x_input, num_iterations, device):
    """
    Run the RNN and collect hidden states at EVERY iteration.
    All computations on GPU.
    
    Args:
        model: The RNN model (on device)
        x_input: Input tensor (B, C_in, H, W)
        num_iterations: Number of RNN iterations
        device: torch.device
    
    Returns:
        trajectory: List of dicts, each containing hidden states at iteration t
    """
    model.eval()
    x_input = x_input.to(device)
    trajectory = []
    
    with torch.no_grad():
        # Initialize x_5 from projection
        x_5 = torch.relu(model.projection(x_input)).to(device)
        
        for t in range(num_iterations):
            # Forward pass through recurrent block
            x_1 = model.recur_block[0](torch.cat([x_5, x_input], dim=1))
            
            conv_x1 = model.recur_block[1][0].conv1(x_1)
            x_21 = torch.relu(conv_x1)
            
            conv_x21 = model.recur_block[1][0].conv2(x_21)
            x_3 = torch.relu(conv_x21 + x_1)
            
            conv_x3 = model.recur_block[1][1].conv1(x_3)
            x_41 = torch.relu(conv_x3)
            
            conv_x41 = model.recur_block[1][1].conv2(x_41)
            x_5 = torch.relu(conv_x41 + x_3)
            
            # Store this iteration's states
            trajectory.append({
                'h1': x_1.clone(),
                'h2': x_21.clone(),
                'h3': x_3.clone(),
                'h4': x_41.clone(),
                'x_out': x_5.clone()
            })
    
    return trajectory

# ============================================================================
# 3. DIRECTION GENERATION
# ============================================================================

def generate_directions(trajectory, device, perturb_layers):
    """
    Generate two orthonormal directions aligned with the trajectory.
    
    Direction 1: The trajectory direction (first -> last state) for perturb_layers only
    Direction 2: A random direction orthogonal to Direction 1
    
    This ensures the trajectory lies IN the 2D plane we're visualizing!
    
    Args:
        trajectory: List of state dicts (single sample)
        device: torch.device
        perturb_layers: List of layer names to perturb
    
    Returns:
        direction1, direction2: Dicts of direction tensors (only for perturb_layers)
        traj_length: The length of the trajectory in hidden state space
    """
    first_state = trajectory[0]
    last_state = trajectory[-1]
    
    # Direction 1: The actual trajectory direction (first -> last) for perturb_layers only
    direction1 = {}
    for key in perturb_layers:
        direction1[key] = (last_state[key] - first_state[key]).to(device)
    
    # Compute norm of direction1 (this is the trajectory length)
    norm1_sq = sum(torch.sum(direction1[key]**2).item() for key in direction1)
    norm1 = np.sqrt(norm1_sq)
    
    if norm1 < 1e-10:
        print("Warning: Trajectory has nearly zero length in perturb_layers!")
        # Fall back to random direction
        for key in perturb_layers:
            direction1[key] = torch.randn_like(first_state[key], device=device)
        norm1_sq = sum(torch.sum(direction1[key]**2).item() for key in direction1)
        norm1 = np.sqrt(norm1_sq)
    
    # Normalize direction1
    for key in direction1:
        direction1[key] = direction1[key] / norm1
    
    # Direction 2: Random direction, orthogonalized against direction1
    direction2 = {}
    for key in perturb_layers:
        direction2[key] = torch.randn_like(first_state[key], device=device)
    
    # Gram-Schmidt: remove component along direction1
    dot_prod = sum(torch.sum(direction1[key] * direction2[key]).item() for key in direction1)
    for key in direction2:
        direction2[key] = direction2[key] - dot_prod * direction1[key]
    
    # Normalize direction2
    norm2_sq = sum(torch.sum(direction2[key]**2).item() for key in direction2)
    norm2 = np.sqrt(norm2_sq)
    if norm2 < 1e-10:
        # If direction2 is degenerate, create a new random one
        for key in perturb_layers:
            direction2[key] = torch.randn_like(first_state[key], device=device)
        norm2_sq = sum(torch.sum(direction2[key]**2).item() for key in direction2)
        norm2 = np.sqrt(norm2_sq)
    
    for key in direction2:
        direction2[key] = direction2[key] / norm2
    
    return direction1, direction2, norm1

# ============================================================================
# 4. COMPUTE TRAJECTORY COORDINATES IN 2D PROJECTION
# ============================================================================

def compute_trajectory_2d_coords(trajectory, reference_state, direction1, direction2, 
                                  perturb_layers, device):
    """
    Project trajectory onto the 2D plane defined by direction1 and direction2.
    Only considers perturb_layers for the projection.
    
    Reference state is the origin (0, 0) - typically the final converged state.
    """
    num_iters = len(trajectory)
    coords_1 = np.zeros(num_iters)
    coords_2 = np.zeros(num_iters)
    
    for t, state in enumerate(trajectory):
        proj1 = 0.0
        proj2 = 0.0
        
        # Only project along the layers we're perturbing
        for key in perturb_layers:
            diff = (state[key].to(device) - reference_state[key].to(device))
            proj1 += torch.sum(diff * direction1[key]).item()
            proj2 += torch.sum(diff * direction2[key]).item()
        
        coords_1[t] = proj1
        coords_2[t] = proj2
    
    return coords_1, coords_2

# ============================================================================
# 5. SAVE/LOAD TRAJECTORY DATA
# ============================================================================

def save_trajectory_data(traj_coords_1, traj_coords_2, traj_losses, 
                          coords, loss_surface, alpha, perturb_layers, 
                          save_dir=None):
    """
    Save trajectory and landscape data to CSV files for later use.
    
    Args:
        traj_coords_1, traj_coords_2: Trajectory coordinates
        traj_losses: Loss at each trajectory point
        coords: Grid coordinates
        loss_surface: 2D loss surface
        alpha: Alpha parameter
        perturb_layers: Layers being perturbed
        save_dir: Directory to save files (defaults to RESULTS_DIR)
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    layer_str = '_'.join(perturb_layers)
    
    # Save trajectory data
    traj_df = pd.DataFrame({
        'iteration': np.arange(len(traj_losses)),
        'coord_1': traj_coords_1,
        'coord_2': traj_coords_2,
        'loss': traj_losses
    })
    traj_path = os.path.join(save_dir, f'trajectory_alpha{alpha}_{layer_str}.csv')
    traj_df.to_csv(traj_path, index=False)
    print(f"Saved trajectory data to: {traj_path}")
    
    # Save loss surface data
    surface_df = pd.DataFrame(loss_surface, index=coords, columns=coords)
    surface_path = os.path.join(save_dir, f'loss_surface_alpha{alpha}_{layer_str}.csv')
    surface_df.to_csv(surface_path)
    print(f"Saved loss surface to: {surface_path}")
    
    # Save metadata
    meta = {
        'alpha': alpha,
        'perturb_layers': perturb_layers,
        'grid_size': len(coords),
        'range_min': coords[0],
        'range_max': coords[-1],
        'num_iterations': len(traj_losses)
    }
    meta_path = os.path.join(save_dir, f'metadata_alpha{alpha}_{layer_str}.csv')
    pd.DataFrame([meta]).to_csv(meta_path, index=False)
    print(f"Saved metadata to: {meta_path}")


def load_trajectory_data(alpha, perturb_layers, load_dir=None):
    """
    Load previously saved trajectory and landscape data.
    
    Returns:
        dict with keys: coords, loss_surface, traj_coords_1, traj_coords_2, traj_losses
    """
    if load_dir is None:
        load_dir = RESULTS_DIR
    
    layer_str = '_'.join(perturb_layers)
    
    # Load trajectory
    traj_path = os.path.join(load_dir, f'trajectory_alpha{alpha}_{layer_str}.csv')
    traj_df = pd.read_csv(traj_path)
    
    # Load surface
    surface_path = os.path.join(load_dir, f'loss_surface_alpha{alpha}_{layer_str}.csv')
    surface_df = pd.read_csv(surface_path, index_col=0)
    
    coords = surface_df.index.values.astype(float)
    loss_surface = surface_df.values
    
    return {
        'coords': coords,
        'loss_surface': loss_surface,
        'traj_coords_1': traj_df['coord_1'].values,
        'traj_coords_2': traj_df['coord_2'].values,
        'traj_losses': traj_df['loss'].values
    }

# ============================================================================
# 6. COMPUTE LOSS LANDSCAPE ON 2D GRID
# ============================================================================

def compute_loss_landscape_with_trajectory(model, x_input, trajectory, alpha,
                                            grid_size=51, range_scale=None,
                                            device='cuda', 
                                            perturb_layers=['h1', 'h2', 'h3', 'h4', 'x_out'],
                                            sample_idx=0):
    """
    Compute loss landscape using trajectory-aligned directions.
    Only perturb_layers are perturbed; other layers stay at reference values.
    """
    device = torch.device(device) if isinstance(device, str) else device
    
    # Use single sample
    x_input_single = x_input[sample_idx:sample_idx+1].to(device)
    
    # Extract single-sample trajectory
    traj_single = [{k: v[sample_idx:sample_idx+1].to(device) for k, v in state.items()} 
                   for state in trajectory]
    
    # Use FINAL state as reference (this will be at origin (0,0))
    reference_state = {k: v.clone() for k, v in traj_single[-1].items()}
    
    # Generate TRAJECTORY-ALIGNED directions (only for perturb_layers)
    # Note: reference_state is not passed since last_state = trajectory[-1] = reference
    print("Computing trajectory-aligned directions...")
    direction1, direction2, traj_length = generate_directions(
        traj_single, device, perturb_layers
    )
    print(f"  Trajectory length (in {perturb_layers} space): {traj_length:.6f}")
    
    # Project trajectory onto 2D plane (only perturb_layers)
    traj_coords_1, traj_coords_2 = compute_trajectory_2d_coords(
        traj_single, reference_state, direction1, direction2, perturb_layers, device
    )
    
    print(f"  Trajectory coord1 range: [{traj_coords_1.min():.6f}, {traj_coords_1.max():.6f}]")
    print(f"  Trajectory coord2 range: [{traj_coords_2.min():.6f}, {traj_coords_2.max():.6f}]")
    print(f"  End point (should be ~0,0): ({traj_coords_1[-1]:.6f}, {traj_coords_2[-1]:.6f})")
    
    # Compute loss at each trajectory point
    print("Computing losses along trajectory...")
    traj_losses = []
    for state in tqdm(traj_single, desc="Trajectory losses"):
        loss = compute_J_alpha(
            model, x_input_single,
            state['h1'], state['h2'], state['h3'], state['h4'], state['x_out'],
            alpha, device
        )
        traj_losses.append(loss.item())
    traj_losses = np.array(traj_losses)
    
    # Determine grid range to include the full trajectory
    if range_scale is None:
        max_coord1 = max(abs(traj_coords_1.min()), abs(traj_coords_1.max()))
        max_coord2 = max(abs(traj_coords_2.min()), abs(traj_coords_2.max()))
        max_range = max(max_coord1, max_coord2)
        range_scale = max(max_range * 1.3, 0.1)  # 30% margin, minimum 0.1
    
    print(f"  Using range_scale = {range_scale:.6f}")
    
    # Create grid
    coords = np.linspace(-range_scale, range_scale, grid_size)
    loss_surface = np.zeros((grid_size, grid_size))
    
    print(f"Computing {grid_size}x{grid_size} loss landscape...")
    
    for i, delta1 in enumerate(tqdm(coords, desc="Computing landscape")):
        for j, delta2 in enumerate(coords):
            # Perturb ONLY perturb_layers from reference state
            perturbed = {}
            for key in ['h1', 'h2', 'h3', 'h4', 'x_out']:
                if key in perturb_layers:
                    perturbed[key] = (reference_state[key] 
                                      + delta1 * direction1[key] 
                                      + delta2 * direction2[key])
                    # Enforce ReLU non-negativity for appropriate layers
                    if key in ['h2', 'h3', 'h4', 'x_out']:
                        perturbed[key] = torch.relu(perturbed[key])
                else:
                    # Keep non-perturbed layers at reference values
                    perturbed[key] = reference_state[key]
            
            loss = compute_J_alpha(
                model, x_input_single,
                perturbed['h1'], perturbed['h2'],
                perturbed['h3'], perturbed['h4'],
                perturbed['x_out'], alpha, device
            )
            loss_surface[i, j] = loss.item()
    
    return coords, loss_surface, traj_coords_1, traj_coords_2, traj_losses

# ============================================================================
# 7. VISUALIZATION WITH TRAJECTORY
# ============================================================================

def plot_landscape_with_trajectory_contour(coords, loss_surface, 
                                            traj_coords_1, traj_coords_2, 
                                            traj_losses,
                                            title="Loss Landscape with RNN Trajectory",
                                            save_path=None, log_scale=True,
                                            num_levels=50):
    """
    Contour plot with trajectory overlaid.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    X, Y = np.meshgrid(coords, coords)
    
    if log_scale:
        # Handle negative values by shifting
        Z = loss_surface
        Z_min = Z.min()
        if Z_min < 0:
            Z_plot = np.log10(Z - Z_min + 1)
        else:
            Z_plot = np.log10(Z + 1)
        clabel = 'log10(J_α + offset)'
    else:
        Z_plot = loss_surface
        clabel = 'J_α'
    
    # Filled contour
    cf = ax.contourf(X, Y, Z_plot.T, levels=num_levels, cmap=cm.viridis)
    
    # Contour lines
    ax.contour(X, Y, Z_plot.T, levels=20, colors='white', linewidths=0.3, alpha=0.5)
    
    # Plot trajectory
    ax.plot(traj_coords_1, traj_coords_2, 'r.-', linewidth=2, markersize=4,
            label='RNN trajectory', alpha=0.8)
    
    # Mark start and end
    ax.plot(traj_coords_1[0], traj_coords_2[0], 'go', markersize=15, 
            label=f'Start (t=0), J={traj_losses[0]:.2e}', zorder=5)
    ax.plot(traj_coords_1[-1], traj_coords_2[-1], 'r*', markersize=20,
            label=f'End (t={len(traj_losses)-1}), J={traj_losses[-1]:.2e}', zorder=5)
    
    # Add iteration numbers at sparse intervals
    num_labels = 10
    step = max(1, len(traj_coords_1) // num_labels)
    for t in range(0, len(traj_coords_1), step):
        ax.annotate(f'{t}', (traj_coords_1[t], traj_coords_2[t]),
                   fontsize=8, color='white',
                   xytext=(3, 3), textcoords='offset points')
    
    ax.set_xlabel('Direction 1', fontsize=12)
    ax.set_ylabel('Direction 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    
    cbar = fig.colorbar(cf, ax=ax, label=clabel)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    
    return fig

def plot_landscape_with_trajectory_3d(coords, loss_surface,
                                       traj_coords_1, traj_coords_2,
                                       traj_losses,
                                       title="3D Loss Landscape with Trajectory",
                                       save_path=None, log_scale=True):
    """
    3D surface plot with trajectory.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(coords, coords)
    
    if log_scale:
        Z = loss_surface
        Z_min = Z.min()
        if Z_min < 0:
            Z_plot = np.log10(Z - Z_min + 1)
            traj_z = np.log10(traj_losses - Z_min + 1)
        else:
            Z_plot = np.log10(Z + 1)
            traj_z = np.log10(traj_losses + 1)
        zlabel = 'log10(J_α + offset)'
    else:
        Z_plot = loss_surface
        traj_z = traj_losses
        zlabel = 'J_α'
    
    # Surface plot
    surf = ax.plot_surface(X, Y, Z_plot.T, cmap=cm.viridis,
                           linewidth=0, antialiased=True, alpha=0.7)
    
    # Trajectory on surface
    ax.plot(traj_coords_1, traj_coords_2, traj_z, 'r-', linewidth=3,
            label='RNN trajectory', zorder=10)
    ax.scatter(traj_coords_1, traj_coords_2, traj_z, c='red', s=20, zorder=10)
    
    # Start and end markers
    ax.scatter([traj_coords_1[0]], [traj_coords_2[0]], [traj_z[0]],
               c='green', s=200, marker='o', label='Start', zorder=11)
    ax.scatter([traj_coords_1[-1]], [traj_coords_2[-1]], [traj_z[-1]],
               c='red', s=300, marker='*', label='End', zorder=11)
    
    ax.set_xlabel('Direction 1', fontsize=12)
    ax.set_ylabel('Direction 2', fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    fig.colorbar(surf, shrink=0.5, aspect=10, label=zlabel)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    
    return fig

def plot_loss_vs_iteration(traj_losses, title="J_α vs RNN Iteration",
                           save_path=None):
    """
    Simple plot of loss value at each iteration.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(len(traj_losses))
    
    ax.semilogy(iterations, np.abs(traj_losses), 'b.-', linewidth=2, markersize=4)
    ax.set_xlabel('RNN Iteration t', fontsize=12)
    ax.set_ylabel('|J_α|', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Mark start and end
    ax.axhline(y=np.abs(traj_losses[-1]), color='r', linestyle='--', 
               label=f'Final loss: {traj_losses[-1]:.2e}')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    
    return fig

# ============================================================================
# 8. MAIN FUNCTION - COMPLETE PIPELINE
# ============================================================================

def visualize_rnn_descent(model, x_input, alpha=2.0, num_iterations=100,
                          grid_size=51, device='cuda', 
                          perturb_layers=['h1', 'h2', 'h3', 'h4', 'x_out'],
                          sample_idx=0,
                          save_data=True,
                          save_prefix=None):
    """
    Complete pipeline: run RNN, compute landscape, visualize with trajectory.
    
    Args:
        model: Trained DTNet model
        x_input: Maze input, shape (B, C_in, H, W) - will use sample_idx
        alpha: Parameter for J_alpha
        num_iterations: Number of RNN iterations
        grid_size: Resolution of landscape grid
        device: Computing device ('cuda' or 'cpu')
        perturb_layers: Which layers to perturb:
            - ['x_out'] for x_out only
            - ['h1', 'h2', 'h3', 'h4', 'x_out'] for all hidden variables
        sample_idx: Which sample in the batch to visualize
        save_data: Whether to save trajectory/landscape data to CSV
        save_prefix: Prefix for saved files (default: auto-generated)
    
    Returns:
        dict with coords, loss_surface, trajectory data
    """
    device = torch.device(device) if isinstance(device, str) else device

    print("="*60)
    print("RNN Loss Landscape Visualization")
    print("="*60)
    print(f"Device: {device}")
    print(f"Alpha: {alpha}")
    print(f"Perturbing layers: {perturb_layers}")
    
    model.eval()
    model = model.to(device)
    x_input = x_input.to(device)
    
    # Collect trajectory
    print(f"\n1. Running RNN for {num_iterations} iterations...")
    trajectory = collect_rnn_trajectory(model, x_input, num_iterations, device)
    print(f"   Collected {len(trajectory)} states")
    
    # Compute landscape with trajectory
    print(f"\n2. Computing loss landscape...")
    coords, loss_surface, traj_c1, traj_c2, traj_losses = \
        compute_loss_landscape_with_trajectory(
            model, x_input, trajectory, alpha,
            grid_size=grid_size, device=device,
            perturb_layers=perturb_layers,
            sample_idx=sample_idx
        )
    
    print(f"\n3. Loss statistics:")
    print(f"   Initial loss (t=0):  {traj_losses[0]:.6e}")
    print(f"   Final loss (t={num_iterations-1}): {traj_losses[-1]:.6e}")
    print(f"   Landscape min: {loss_surface.min():.6e}")
    print(f"   Landscape max: {loss_surface.max():.6e}")
    
    # Save data
    if save_data:
        print(f"\n4. Saving data to CSV...")
        save_trajectory_data(traj_c1, traj_c2, traj_losses, 
                              coords, loss_surface, alpha, perturb_layers)
        
    # Generate plots
    print(f"\n5. Generating visualizations...")

    layer_str = '_'.join(perturb_layers)
    if save_prefix is None:
        save_prefix = os.path.join(RESULTS_DIR, f'landscape_alpha{alpha}_{layer_str}')
    
    # Contour plot with trajectory
    plot_landscape_with_trajectory_contour(
        coords, loss_surface, traj_c1, traj_c2, traj_losses,
        title=f'J_α Loss Landscape with RNN Trajectory\n(α={alpha}, layers: {layer_str})',
        save_path=f'{save_prefix}_contour.png',
        log_scale=True
    )
    
    # 3D plot with trajectory
    plot_landscape_with_trajectory_3d(
        coords, loss_surface, traj_c1, traj_c2, traj_losses,
        title=f'3D J_α Landscape with Trajectory\n(α={alpha}, layers: {layer_str})',
        save_path=f'{save_prefix}_3d.png',
        log_scale=True
    )
    
    # Loss vs iteration
    plot_loss_vs_iteration(
        traj_losses,
        title=f'J_α Convergence (α={alpha})',
        save_path=f'{save_prefix}_convergence.png'
    )
    
    print(f"\n6. Done! Results saved in '{RESULTS_DIR}'")
    
    return {
        'coords': coords,
        'loss_surface': loss_surface,
        'traj_coords_1': traj_c1,
        'traj_coords_2': traj_c2,
        'traj_losses': traj_losses,
        'trajectory': trajectory
    }

def plot_from_saved_data(alpha, perturb_layers, load_dir=None, save_prefix=None):
    """
    Load saved data and regenerate plots without recomputing.
    
    Args:
        alpha: Alpha value used when saving
        perturb_layers: Layers used when saving
        load_dir: Directory to load from (defaults to RESULTS_DIR)
        save_prefix: Prefix for new saved figures
    """
    print("Loading saved data...")
    data = load_trajectory_data(alpha, perturb_layers, load_dir)
    
    layer_str = '_'.join(perturb_layers)
    if save_prefix is None:
        save_prefix = os.path.join(RESULTS_DIR, f'landscape_alpha{alpha}_{layer_str}')
    
    # Regenerate plots
    plot_landscape_with_trajectory_contour(
        data['coords'], data['loss_surface'], 
        data['traj_coords_1'], data['traj_coords_2'], data['traj_losses'],
        title=f'J_α Loss Landscape with RNN Trajectory\n(α={alpha}, layers: {layer_str})',
        save_path=f'{save_prefix}_contour.png',
        log_scale=True
    )
    
    plot_landscape_with_trajectory_3d(
        data['coords'], data['loss_surface'],
        data['traj_coords_1'], data['traj_coords_2'], data['traj_losses'],
        title=f'3D J_α Landscape with Trajectory\n(α={alpha}, layers: {layer_str})',
        save_path=f'{save_prefix}_3d.png',
        log_scale=True
    )
    
    plot_loss_vs_iteration(
        data['traj_losses'],
        title=f'J_α Convergence (α={alpha})',
        save_path=f'{save_prefix}_convergence.png'
    )
    
    return data

# ============================================================================
# 8. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage - run from the project root directory:
        python loss_landscape/visualize.py
    """
    
    import sys
    # Add parent directory to path to import deepthinking
    sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
    
    from deepthinking.models import dt_net_recall_2d
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    
    # Paths relative to project root (one level up from loss_landscape/)
    PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'outputs/mazes_ablation/training-rusty-Tayla/model_best.pth')
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data/maze_data_test_9')
    
    ALPHA = 2.0
    NUM_ITERATIONS = 30
    GRID_SIZE = 21  # Use 21 for quick test, 51-101 for publication quality
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Results will be saved to: {RESULTS_DIR}")

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    print("\nLoading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    model = dt_net_recall_2d(width=128, in_channels=3, max_iters=NUM_ITERATIONS)
    
    state_dict = checkpoint['net']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # Load maze data
    # -------------------------------------------------------------------------
    print("Loading maze data...")
    x_np = np.load(os.path.join(DATA_PATH, 'inputs.npy'))
    
    example_idx = 0
    x_input = torch.tensor(x_np[example_idx:example_idx+1], dtype=torch.float32)
    x_input = x_input.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    
    print(f"Input shape: {x_input.shape}")

    # -------------------------------------------------------------------------
    # Case 1: Perturb only x_out
    # -------------------------------------------------------------------------
    # print("\n" + "="*60)
    # print("CASE 1: Perturbing x_out only")
    # print("="*60)
    
    # results_xout = visualize_rnn_descent(
    #     model=model,
    #     x_input=x_input,
    #     alpha=ALPHA,
    #     num_iterations=NUM_ITERATIONS,
    #     grid_size=GRID_SIZE,
    #     device=device,
    #     perturb_layers=['x_out'],  # Only x_out
    #     sample_idx=0,
    #     save_data=True
    # )
    
    # -------------------------------------------------------------------------
    # Case 2: Perturb all hidden variables
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("CASE 2: Perturbing all hidden variables")
    print("="*60)
    
    results_all = visualize_rnn_descent(
        model=model,
        x_input=x_input,
        alpha=ALPHA,
        num_iterations=NUM_ITERATIONS,
        grid_size=GRID_SIZE,
        device=device,
        perturb_layers=['h1', 'h2', 'h3', 'h4', 'x_out'],  # All layers
        sample_idx=0,
        save_data=True
    )
    
    # -------------------------------------------------------------------------
    # Example: Reload and plot from saved data
    # -------------------------------------------------------------------------
    # print("\n" + "="*60)
    # print("Reloading and plotting from saved CSV data...")
    # print("="*60)
    
    # This demonstrates how to regenerate plots without recomputing
    # data_reloaded = plot_from_saved_data(
    #     alpha=ALPHA, 
    #     perturb_layers=['x_out']
    # )
    
    # print("\nAll done!")
