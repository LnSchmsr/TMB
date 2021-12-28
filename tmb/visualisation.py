from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px
import pandas as pd
import itertools


def plt_a(
    acceleration: np.ndarray[np.float64],
    filepath: str = "",
    headline: str = "",
    labels: List[str] = None,
    ylabel: str = "a [m/s²]",
    xlabel: str = "Time steps",
):
    """
    Plots multiple acceleration signals.

    Args:
        acceleration (np.array[np.float64]): Acceleration signal: [Signal(s), Wheel(s)]
        filepath (str, optional): Filepath for saving. Defaults to ''.
        headline (str, optional): Headline. Defaults to ''.
        labels (List[str], optional): HUE for each signal. Defaults to None.
        xlabel (str, optional): Label of x-axis. Defaults to Time steps.
        ylabel (str, optional): Label of y-axis. Defaults to a [m/s²].
    """
    df = pd.DataFrame(acceleration, columns=labels)
    fig = px.line(
        df, labels=dict(index=xlabel, value=ylabel, variable="Wheels"), title=headline
    )

    if filepath !=  "":
        fig.write_image(filepath + ".pdf")
    fig.show()


def plt_subplots_a(
    acceleration: np.ndarray[np.float64],
    filepath: str = "",
    headline: str = "",
    labels: List[str] = None,
    ylabel: str = "a [m/s²]",
    xlabel: str = "Time steps",
):
    """
    Plots multiple acceleration signals.

    Args:
        acceleration (np.ndarray[np.float64]): Acceleration signal: [Signal(s), Wheel(s)]
        filepath (str, optional): Filepath for saving. Defaults to ''.
        headline (str, optional): Headline. Defaults to ''.
        labels (List[str], optional): HUE for each signal. Defaults to None.
        xlabel (str, optional): Label of x-axis. Defaults to Time steps.
        ylabel (str, optional): Label of y-axis. Defaults to a [m/s²].
    """
    rows = acceleration.shape[1] if acceleration.ndim == 2 else len(acceleration)
    fig = make_subplots(rows=rows, cols=1, x_title=xlabel, y_title=ylabel)
    for i, a in enumerate(acceleration.T, start=1):
        df = pd.DataFrame(a)
        fig.add_trace(
            go.Scatter(
                x=df[0].index,
                y=df[0].values,
                name=labels[i - 1] if labels is not None else None,
            ),
            row=i,
            col=1,
        )

    fig.update_layout(title_text=headline)
    if filepath != "":
        fig.write_image(filepath + ".pdf")
    fig.show()


def plt_a_3d(
    acceleration: np.ndarray[np.float64],
    plot_2d: bool = False,
    show_summed: bool = False,
):
    """
    Plots a dataset in 3D

    Args:
        acceleration (np.ndarray[np.float64]): Input Matrix: [Signals, Wheels]
        plot_2d (bool, optional): If true plots 2d. Defaults to False.
        show_summed (bool, optional): If true plots the summed signal. Defaults to False.
    """
    x_max = len(acceleration)
    datasets = [
        [
            np.full(shape=(len(acceleration[i])), fill_value=i).flatten().tolist(),
            np.arange(0, len(acceleration[i]), 1).tolist(),
            acceleration[i].flatten().tolist(),
            np.full(shape=(len(acceleration[i])), fill_value=i + 1).tolist(),
        ]
        for i in range(x_max)
    ]
    if show_summed:
        lngth = len(acceleration[-1])
        accel = [np.resize(x, lngth) for x in acceleration]
        acc_summed = np.vstack(accel).sum(axis=0)
        datasets.append(
            [
                np.full(shape=(lngth), fill_value=x_max + 1).flatten().tolist(),
                np.arange(0, lngth, 1).tolist(),
                acc_summed.flatten().tolist(),
                np.full(shape=(lngth), fill_value=x_max + 1).tolist(),
            ]
        )
    dataset_tuples = [list(zip(*data)) for data in datasets]
    dataset_tuples = list(itertools.chain.from_iterable(dataset_tuples))
    df = pd.DataFrame(data=dataset_tuples, columns=["x", "y", "z", "wheel"])
    fig = (
        px.line_3d(data_frame=df, x="x", y="y", z="z", color="wheel")
        if not plot_2d
        else px.line(data_frame=df, x="y", y="z", color="wheel")
    )
    fig.show()
