"""
A series of helper functions
"""
import os
import torch
import numpy as np
import plotly.graph_objects as go


def walk_through_dir(dir_path,file_type='*'):
    """
    Walks through dir_path using os.walk returning its contents.
    Args:
        dir_path (str): target directory
        file_type (str): filter files as per given string ex '.jpg'

    Returns:
    A print out of:
        number of subdiretories in dir_path
        number of files in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirname, filename in os.walk(dir_path):
        if file_type == '*':
            print(f"There are {len(dirname)} directories and {len(filename)} files in '{dirpath}'.")
        elif file_type in filename:
            print(f"There are {len(dirname)} directories and {len(filename)} files in '{dirpath}'.")

def plot_decision_boundary(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on x in comparison to y.
    Source - https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py#L76
    Args:
        model (torch.nn.Module): target model
        x (torch.Tensor): input space
        y (torch.Tensor): output space (logits)
    Returns:
        A contour plot with scattered predictions
    Notes: use plotly
    """

    # Setup prediction boundaries and grid
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Make features (convert grid to x,y coords matrix)
    x_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(x_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Plot the graph
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xx.ravel(),
        y=yy.ravel(),
        z=y_pred.ravel(),
        opacity=0.5,
        name='contour',
        colorscale='Plasma'
        ))
    fig.add_trace(
      go.Scatter(
        x=x[:, 0],
        y=x[:, 1],
        mode='markers',
        name="data",
        marker = dict(
          size = 10,
          color = y,
          colorscale='Plasma'
      ))
    )
    fig.update_layout(title='Plot descisioin boundry',
                      showlegend=False,
                      width=600,
                      height=600)

    fig.show()

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc