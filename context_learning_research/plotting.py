from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch.nn.functional as F
import torch


def plot_sequence_prediction(factual_seq: list,
                             predicted_seq: list,
                             end_of_learning_seq: Optional[int] = None,
                             value_name: str = "Token's Index",
                             factual_name: str = "Factual",
                             predicted_name: str = "Predicted",
                             show_range: Optional[tuple] = None,
                             factual_values_space_padding: int = 10,
                             factual_marker_size: int = 10,
                             predicted_marker_size: int = 7,
                             height: int = 400, width: int = 1200,
                             save_plot_as: Optional[str] = None):
    fig = go.Figure()
    start_from = show_range[0] if show_range else 0
    end_on = show_range[1] + 1 if show_range else len(factual_seq) + 1
    x = np.arange(start_from + 2, len(factual_seq) + 1)

    if factual_seq:
        fig.add_trace(go.Scatter(x=x, y=factual_seq[start_from:end_on], mode='markers', marker=dict(size=factual_marker_size), name=factual_name))
    if predicted_seq:
        fig.add_trace(go.Scatter(x=x, y=predicted_seq[start_from:end_on], mode='markers', marker=dict(size=predicted_marker_size), name=predicted_name))
    if end_of_learning_seq:
        fig.add_vline(x=end_of_learning_seq, line_width=4, line_dash="dash", line_color="green")
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name="End of context for pattern learning"))
    min_value = max(np.min(factual_seq) - factual_values_space_padding, 0)
    max_value = np.max(factual_seq) + factual_values_space_padding

    fig.update_layout(height=height, width=width, margin=dict(t=50, b=50, l=50, r=50))
    fig.update_xaxes(title_text="Element's Index", range=[start_from, end_on + 1])
    fig.update_yaxes(title_text=value_name, range=[min_value, max_value])
    if save_plot_as:
        fig.write_html(save_plot_as)
    else:
        fig.show()


def plot_log_prob(logits: list,
                 factual_seq: list,
                 noise_indices: Optional[list] = None,
                 end_of_learning_seq: Optional[int] = None,
                 factual_name: str = "Factual",
                 predicted_name: str = "Predicted",
                 show_range: Optional[tuple] = None,
                 factual_line_width: int = 3,
                 predicted_line_width: int = 5,
                 height: int = 400, width: int = 1200,
                 save_plot_as: Optional[str] = None):
    fig = go.Figure()
    start_from = show_range[0] if show_range else 0
    end_on = show_range[1] + 1 if show_range else len(factual_seq) + 1
    x = np.arange(start_from + 2, len(factual_seq) + 1)

    logits = torch.Tensor(logits)
    log_probs = F.log_softmax(logits, dim=-1)
    actual_log_probs = log_probs[torch.arange(logits.shape[0]), factual_seq].numpy()
    predicted_log_probs = log_probs.numpy().max(axis=1)

    fig.add_trace(go.Scatter(x=x, y=predicted_log_probs[start_from:end_on], mode='lines', line={'width': predicted_line_width}, name=predicted_name))
    fig.add_trace(go.Scatter(x=x, y=actual_log_probs[start_from:end_on], mode='lines', line={'width': factual_line_width}, name=factual_name))
    if end_of_learning_seq:
        fig.add_vline(x=end_of_learning_seq, line_width=4, line_dash="dash", line_color="green")
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name="End of context for pattern learning"))
    if noise_indices:
        for noise_index in noise_indices:
            fig.add_vline(x=noise_index, line_width=factual_line_width, line_dash="dash", line_color="red")
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name="Elements indices at which noise was added"))

    fig.update_layout(height=height, width=width, margin=dict(t=50, b=50, l=50, r=50))
    fig.update_xaxes(title_text="Element's Index", range=[start_from, end_on])
    fig.update_yaxes(title_text="Log-Probability")
    if save_plot_as:
        fig.write_html(save_plot_as)
    else:
        fig.show()