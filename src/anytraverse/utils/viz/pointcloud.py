import torch
import plotly.graph_objects as go
import numpy as np

def plot_pointcloud_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    title: str = "Pointcloud",
    color_intensity: torch.Tensor | None = None,
    sample_size: int = 10_000,
) -> go.Figure:
    """Creates an interactive 3D visualization of a point cloud using Plotly.

    This function generates a three-dimensional scatter plot of points defined by their
    x, y, and z coordinates. Points can be colored either based on their z-coordinate
    values (default) or using custom color intensities. If the number of points exceeds
    the specified sample_size, a random subset of points will be plotted.

    Params:
        x (torch.Tensor): x-coordinates of the points. Shape (N,) where N is number of points
        y (torch.Tensor): y-coordinates of the points. Shape (N,) where N is number of points
        z (torch.Tensor): z-coordinates of the points. Shape (N,) where N is number of points
        color_intensity (torch.Tensor, optional): Values used to color the points. Shape (N,)
            where N is number of points. If None, points colored by z-coordinates
        sample_size (int, optional): Maximum number of points to plot. If input contains more
            points, a random subset will be sampled. Defaults to 10000

    Returns:
        go.Figure: Plotly figure with interactive 3D visualization including:
            - 3D scatter plot with points colored by z-coordinate or color_intensity
            - Axis labels and title
            - Colorbar showing the color scale
            - Interactive controls for rotation, zoom, and pan

    Raises:
        ValueError: If x, y, and z tensors have different lengths, or if color_intensity
            is provided with different length than coordinate tensors
        TypeError: If inputs are not torch.Tensor objects

    Example:
        >>> # Basic usage with z-axis coloring
        >>> fig = plot_point_cloud_3d(x_coords, y_coords, z_coords)
        >>> # Usage with custom color intensities
        >>> custom_colors = torch.rand(1000)  # Random color values
        >>> fig = plot_point_cloud_3d(x_coords, y_coords, z_coords,
        ...                          color_intensity=custom_colors)
    """
    # Input validation
    if not all(isinstance(t, torch.Tensor) for t in [x, y, z]):
        raise TypeError("Coordinates must be torch.Tensor objects")

    if not (x.shape == y.shape == z.shape):
        raise ValueError("Coordinate tensors must have the same shape")

    if color_intensity is not None:
        if not isinstance(color_intensity, torch.Tensor):
            raise TypeError("color_intensity must be a torch.Tensor")
        if color_intensity.shape != x.shape:
            raise ValueError(
                "color_intensity must have the same shape as coordinate tensors"
            )

    # Sample points if the point cloud is too large
    if x.shape[0] > sample_size:
        indices = torch.randperm(x.shape[0])[:sample_size]
        x, y, z = x[indices], y[indices], z[indices]
        if color_intensity is not None:
            color_intensity = color_intensity[indices]

    # Convert to numpy for plotly
    x_plot, y_plot, z_plot = x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy()

    # Determine color values
    color_values = (
        color_intensity.cpu().numpy() if color_intensity is not None else z_plot
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_plot,
                y=y_plot,
                z=z_plot,
                mode="markers",
                marker=dict(
                    size=2,
                    color=color_values,
                    colorscale="Jet",
                    opacity=0.8,
                    showscale=True,  # Add colorbar
                    colorbar=dict(
                        title="Color Intensity"
                        if color_intensity is not None
                        else "Z-axis"
                    ),
                ),
            ),

            #plot 3D plane given parameters

        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=800,
        height=800,
    )

    return fig



def plot_pointcloud_3d_with_plane(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    a: float,
    b: float,
    c: float,
    d: float,
    title: str = "Pointcloud",
    color_intensity: torch.Tensor | None = None,
    sample_size: int = 10_000,
) -> go.Figure:
    """Creates an interactive 3D visualization of a point cloud using Plotly.

    This function generates a three-dimensional scatter plot of points defined by their
    x, y, and z coordinates. Points can be colored either based on their z-coordinate
    values (default) or using custom color intensities. If the number of points exceeds
    the specified sample_size, a random subset of points will be plotted.

    Params:
        x (torch.Tensor): x-coordinates of the points. Shape (N,) where N is number of points
        y (torch.Tensor): y-coordinates of the points. Shape (N,) where N is number of points
        z (torch.Tensor): z-coordinates of the points. Shape (N,) where N is number of points
        color_intensity (torch.Tensor, optional): Values used to color the points. Shape (N,)
            where N is number of points. If None, points colored by z-coordinates
        sample_size (int, optional): Maximum number of points to plot. If input contains more
            points, a random subset will be sampled. Defaults to 10000

    Returns:
        go.Figure: Plotly figure with interactive 3D visualization including:
            - 3D scatter plot with points colored by z-coordinate or color_intensity
            - Axis labels and title
            - Colorbar showing the color scale
            - Interactive controls for rotation, zoom, and pan

    Raises:
        ValueError: If x, y, and z tensors have different lengths, or if color_intensity
            is provided with different length than coordinate tensors
        TypeError: If inputs are not torch.Tensor objects

    Example:
        >>> # Basic usage with z-axis coloring
        >>> fig = plot_point_cloud_3d(x_coords, y_coords, z_coords)
        >>> # Usage with custom color intensities
        >>> custom_colors = torch.rand(1000)  # Random color values
        >>> fig = plot_point_cloud_3d(x_coords, y_coords, z_coords,
        ...                          color_intensity=custom_colors)
    """
    # Input validation
    if not all(isinstance(t, torch.Tensor) for t in [x, y, z]):
        raise TypeError("Coordinates must be torch.Tensor objects")

    if not (x.shape == y.shape == z.shape):
        raise ValueError("Coordinate tensors must have the same shape")

    if color_intensity is not None:
        if not isinstance(color_intensity, torch.Tensor):
            raise TypeError("color_intensity must be a torch.Tensor")
        if color_intensity.shape != x.shape:
            raise ValueError(
                "color_intensity must have the same shape as coordinate tensors"
            )

    # Sample points if the point cloud is too large
    if x.shape[0] > sample_size:
        indices = torch.randperm(x.shape[0])[:sample_size]
        x, y, z = x[indices], y[indices], z[indices]
        if color_intensity is not None:
            color_intensity = color_intensity[indices]

    # Convert to numpy for plotly
    x_plot, y_plot, z_plot = x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy()

    # Determine color values
    color_values = (
        color_intensity.cpu().numpy() if color_intensity is not None else z_plot
    )

    # Create a grid of x, y coordinates
    x = np.linspace(min(x_plot), max(x_plot), 500)
    y = np.linspace(min(y_plot), min(y_plot), 500)
    X, Y = np.meshgrid(x, y)
    
    # Calculate z coordinates from the plane equation: ax + by + cz = d
    # Rearranging to: z = (d - ax - by)/c
    if c != 0:
        Z = (d - a*X - b*Y) / c
    else:
        raise ValueError("Coefficient 'c' cannot be zero for explicit z representation")
    
    # Create the 3D surface plot
    # fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_plot,
                y=y_plot,
                z=z_plot,
                mode="markers",
                marker=dict(
                    size=2,
                    color=color_values,
                    colorscale="Jet",
                    opacity=0.8,
                    showscale=True,  # Add colorbar
                    colorbar=dict(
                        title="Color Intensity"
                        if color_intensity is not None
                        else "Z-axis"
                    ),
                ),
            ),

            #plot 3D plane given parameters
            go.Surface(x=x, y=y, z=Z)
        ]
    )

    print(len(Z))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        # width=800,
        # height=800,
    )

    return fig
