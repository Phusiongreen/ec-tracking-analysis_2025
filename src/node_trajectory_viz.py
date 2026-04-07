"""
3D Visualization of Node Trajectory Through Time

This module provides visualization tools for tracking individual nodes (cells)
through time in the Delaunay graph structure. The visualization shows:
- The tracked node's 3D trajectory (X, Y spatial coordinates + Z time axis)
- The node's neighbors at each timepoint (semi-transparent)
- Summary statistics about the node's motion
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_node_trajectory_3d(
    node_id,
    graphs,
    experimentID,
    observation_time,
    figsize=(12, 9),
    node_color='red',
    node_size=100,
    neighbor_color='lightblue',
    neighbor_size=50,
    neighbor_alpha=0.4,
    trajectory_linewidth=2,
    trajectory_color='red',
    show_neighbors=False,
    save_path=None
):
    """
    Visualize a single node's trajectory through time in 3D space.
    
    Creates a 3D plot where:
    - X and Y axes show spatial coordinates (μm)
    - Z axis shows time (frames)
    - The tracked node is shown as a red line with markers
    - Neighbor nodes are shown as semi-transparent blue dots at each timepoint
    
    Parameters:
    -----------
    node_id : int
        The TRACK_ID of the node to track
    graphs : dict
        Dictionary with experimentID keys containing list of graphs (one per timepoint).
        Graphs should be networkx Graph objects with node attributes 'x' and 'y'.
    experimentID : str or int
        The experiment ID to extract graphs from
    observation_time : tuple
        (start_frame, end_frame) tuple defining the time range
    figsize : tuple, optional
        Figure size for the plot (default: 12x9)
    node_color : str, optional
        Color for the tracked node (default: 'red')
    node_size : int, optional
        Size for the tracked node markers (default: 100)
    neighbor_color : str, optional
        Color for neighbor nodes (default: 'lightblue')
    neighbor_size : int, optional
        Size for neighbor markers (default: 50)
    neighbor_alpha : float, optional
        Transparency for neighbor nodes 0-1 (default: 0.4)
    trajectory_linewidth : float, optional
        Width of the trajectory line (default: 2)
    trajectory_color : str, optional
        Color of the trajectory line (default: 'red')
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    
    Raises:
    -------
    ValueError
        If node_id is not found in any timepoint of the specified experiment
    
    Examples:
    ---------
    >>> # Basic usage
    >>> fig, ax = plot_node_trajectory_3d(
    ...     node_id=123,
    ...     graphs=graphs,
    ...     experimentID='exp_001',
    ...     observation_time=(0, 100)
    ... )
    >>> plt.show()
    
    >>> # Customized visualization
    >>> fig, ax = plot_node_trajectory_3d(
    ...     node_id=456,
    ...     graphs=graphs,
    ...     experimentID='exp_001',
    ...     observation_time=(0, 100),
    ...     figsize=(14, 10),
    ...     neighbor_alpha=0.35,
    ...     trajectory_linewidth=2.5,
    ...     save_path='node_trajectory.png'
    ... )
    """
    
    # Extract the list of graphs for this experiment
    t_graphs = graphs[experimentID]
    start_frame, end_frame = observation_time
    
    # Collect node positions over time
    node_trajectory = []
    timepoints = []
    neighbors_per_t = []
    
    for t_idx, t in enumerate(range(start_frame, end_frame)):
        G = t_graphs[t_idx]
        
        # Check if node exists in this timepoint's graph
        if node_id in G.nodes():
            x = G.nodes[node_id].get('x', None)
            y = G.nodes[node_id].get('y', None)
            
            if x is not None and y is not None:
                node_trajectory.append([x, y, t])
                timepoints.append(t)

                if show_neighbors:
                    # Get neighbors of this node
                    neighbors = list(G.neighbors(node_id))
                    neighbors_coords = []
                    for neighbor_id in neighbors:
                        neighbor_x = G.nodes[neighbor_id].get('x', None)
                        neighbor_y = G.nodes[neighbor_id].get('y', None)
                        if neighbor_x is not None and neighbor_y is not None:
                            neighbors_coords.append([neighbor_x, neighbor_y, t])

                    neighbors_per_t.append(neighbors_coords)
            else:
                neighbors_per_t.append([])
        else:
            neighbors_per_t.append([])
    
    # Check if we found any data for this node
    if len(node_trajectory) == 0:
        print(f"Warning: Node {node_id} not found in any timepoint!")
        return None, None
    
    node_trajectory = np.array(node_trajectory)
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory line
    ax.plot(
        node_trajectory[:, 0],
        node_trajectory[:, 1],
        node_trajectory[:, 2],
        color=trajectory_color,
        linewidth=trajectory_linewidth,
        label=f'Node {node_id} trajectory',
        alpha=0.8
    )
    
    # Plot the main node at each timepoint
    ax.scatter(
        node_trajectory[:, 0],
        node_trajectory[:, 1],
        node_trajectory[:, 2],
        c=node_color,
        s=node_size,
        marker='o',
        edgecolors='darkred',
        linewidth=1.5,
        label=f'Node {node_id}',
        zorder=5
    )
    
    # Plot neighbors at each timepoint (semi-transparent)
    # Collect all neighbor coordinates first to plot them all at once
    all_neighbors = []
    for neighbors_coords in neighbors_per_t:
        if len(neighbors_coords) > 0:
            all_neighbors.extend(neighbors_coords)
    
    if len(all_neighbors) > 0:
        all_neighbors_array = np.array(all_neighbors)
        ax.scatter(
            all_neighbors_array[:, 0],
            all_neighbors_array[:, 1],
            all_neighbors_array[:, 2],
            c=neighbor_color,
            s=neighbor_size,
            marker='o',
            alpha=neighbor_alpha,
            edgecolors='none',
            depthshade=False,
            zorder=1
        )
    
    # Labels and formatting
    ax.set_xlabel('X position (μm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y position (μm)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Frame (time)', fontsize=11, fontweight='bold')
    ax.set_title(
        f'Node {node_id} Trajectory Over Time\nExperiment {experimentID}',
        fontsize=13,
        fontweight='bold',
        pad=20
    )
    
    # Set z-axis to frame range
    ax.set_zlim(start_frame, end_frame)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Add grid without panes to avoid visual clutter
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Adjust viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Print summary statistics
    print(f"\n--- Node {node_id} Trajectory Summary ---")
    print(f"Number of timepoints: {len(node_trajectory)}")
    print(f"Time range (frames): {int(timepoints[0])} to {int(timepoints[-1])}")
    print(f"X range (μm): {node_trajectory[:, 0].min():.1f} to {node_trajectory[:, 0].max():.1f}")
    print(f"Y range (μm): {node_trajectory[:, 1].min():.1f} to {node_trajectory[:, 1].max():.1f}")
    
    # Calculate displacement
    start_pos = node_trajectory[0, :2]
    end_pos = node_trajectory[-1, :2]
    displacement = np.linalg.norm(end_pos - start_pos)
    print(f"Total displacement (μm): {displacement:.1f}")

    # Calculate average neighbors (only if we have neighbor data)
    neighbor_counts = [len(n) for n in neighbors_per_t]
    if neighbor_counts:
        avg_neighbors = np.mean(neighbor_counts)
        print(f"Average neighbors per frame: {avg_neighbors:.1f}")
    else:
        print(f"Average neighbors per frame: N/A (show_neighbors=False)")
    print("-----------------------------------\n")
    return fig, ax

def plot_multiple_nodes_3d(
    node_ids,
    graphs,
    experimentID,
    observation_time,
    figsize=(14, 10),
    colors=None,
    trajectory_linewidth=1.5,
    save_path=None
):
    """
    Visualize multiple nodes' trajectories through time in a single 3D plot.
    
    Parameters:
    -----------
    node_ids : list of int
        List of TRACK_IDs to track
    graphs : dict
        Dictionary with experimentID keys containing list of graphs
    experimentID : str or int
        The experiment ID to extract graphs from
    observation_time : tuple
        (start_frame, end_frame) tuple
    figsize : tuple, optional
        Figure size (default: 14x10)
    colors : list of str, optional
        List of colors for each node (defaults to matplotlib color cycle)
    neighbor_alpha : float, optional
        Transparency for neighbor nodes (default: 0.2)
    trajectory_linewidth : float, optional
        Width of trajectory lines (default: 1.5)
    save_path : str, optional
        If provided, save figure to this path
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    t_graphs = graphs[experimentID]
    start_frame, end_frame = observation_time
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(node_ids)))
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    for idx, node_id in enumerate(node_ids):
        node_trajectory = []
        
        for t_idx, t in enumerate(range(start_frame, end_frame)):
            G = t_graphs[t_idx]
            
            if node_id in G.nodes():
                x = G.nodes[node_id].get('x', None)
                y = G.nodes[node_id].get('y', None)
                if x is not None and y is not None:
                    node_trajectory.append([x, y, t])
        
        if len(node_trajectory) > 0:
            node_trajectory = np.array(node_trajectory)
            color = colors[idx] if isinstance(colors[idx], str) else colors[idx]
            
            # Plot trajectory line
            ax.plot(
                node_trajectory[:, 0],
                node_trajectory[:, 1],
                node_trajectory[:, 2],
                color=color,
                linewidth=trajectory_linewidth,
                label=f'Node {node_id}',
                alpha=0.8
            )
            
            # Plot nodes
            ax.scatter(
                node_trajectory[:, 0],
                node_trajectory[:, 1],
                node_trajectory[:, 2],
                c=[color],
                s=50,
                marker='o',
                edgecolors='black',
                linewidth=0.5,
                zorder=5
            )
    
    ax.set_xlabel('X position (μm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y position (μm)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Frame (time)', fontsize=11, fontweight='bold')
    ax.set_title(
        f'Multiple Node Trajectories Over Time\nExperiment {experimentID}',
        fontsize=13,
        fontweight='bold',
        pad=20
    )
    
    ax.set_zlim(start_frame, end_frame)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax

