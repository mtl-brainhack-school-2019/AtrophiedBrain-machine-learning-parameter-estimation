# Machine Learning Parameter Estimation

## Summary
The structures of the brain change naturally with aging and diseases modify the brain in their own way. Increased knowledge of the location and timing of the changes caused by disease could teach us about how diseases progress. This project used machine learning to model how the brain naturally changes.

## Project definition
Unfortunately, we do not fully understand how the brain normally changes. A model of how the structures of the brain change in normal aging could serve as a baseline for comparison with models of specific diseases. Neural networks are proving to be useful for many scientific tasks, can they be useful for mathematical modeling?

Ordinary differential equations (ODEs) are commonly used to model biological systems. A [recent paper](https://arxiv.org/abs/1806.07366) introduced the concept of Neural Ordinary Differential Equations that reportedly allow ODE models to be trained using deep learning tools. This project explores the use of Neural ODEs for estimating the parameters of an ODE-based model of the change of grey matter in the brain.

A simple single-parameter model of Cortical Thickness was chosen as the model to be studied using Neural ODEs. This single parameter is effectively the rate of atrophy or growth of each region of the brain. Click HERE for an interactive demo of this model.

## Learning experience
This project primarily used [Python](https://www.python.org/), [PyTorch](https://pytorch.org/) (Python’s deep learning framework), and [Jupyter Notebooks](https://jupyter.org/) stored in a public GitHub repository. [Dash by Plotly](https://plot.ly/dash/) was used to make interactive visualizations. [Julia](https://julialang.org/) and Julia’s deep learning library [Flux.jl](https://fluxml.ai/) were also used, as their Neural ODE library [DiffEqFlux](https://github.com/JuliaDiffEq/DiffEqFlux.jl) is under heavy development (click [here](https://www.youtube.com/watch?v=5ZgEp36E71Y&amp=&index=37&amp=&t=0s) for an introduction).

As a first task, the [PyTorch Neural ODE GitHub repository](https://github.com/rtqichen/torchdiffeq) provided by the authors of [the NIPS Neural ODE paper](https://arxiv.org/abs/1806.07366) was used to predict simulated cortical thickness trajectories. Three different models were used and it was interesting to note the different ability of these models to converge to the solution.

<img src="https://github.com/mtl-brainhack-school-2019/AtrophiedBrain-machine-learning-parameter-estimation/raw/master/figures/all_3_011.PNG" width=800>

The figure above illustrates the three models tested and how well they converged after approximately 120 training epochs. The left model used a traditional feed-forward neural network, the center model used a derivative layer that calculated the derivative of interest, and the right model combined the derivative layer with a feed-forward neural network. The most noteworthy take away from this experiment was that combining knowledge of the model of interest (in the form of the derivative layer) enabled the feed-forward neural network to converge must faster than the feed-forward model could on its own. Also interesting is that the derivative layer alone, without a feed-forward network, was unable to converge even after thousands of training epochs. Higher resolution examples of these networks and their training graphs are available HERE.

## Results
The experiment described above showed the usefulness of using Neural ODEs for a regression task; however, they did not provide the model parameter which was the goal of this experiment. The [Neural ODE paper](https://arxiv.org/abs/1806.07366) presented the following figure which inspired a second experiment where a generative neural network would be used to extract the parameter(s) of interest in the first half of the network, followed by a model that attempted to reproduce the input trajectories using only the model parameters and the initial cortical thickness values.

<img src="https://github.com/mtl-brainhack-school-2019/AtrophiedBrain-machine-learning-parameter-estimation/raw/master/figures/generator_nn.PNG">

The first half of this model was created using [DiffEqFlux](https://github.com/JuliaDiffEq/DiffEqFlux.jl). This model was trained on thousands of simulated cortical thickness trajectories and could output a prediction of the model's parameter. Unfortunately, insufficient time was available to create the complete model.

The deliverables for this project are an interactive Jupyter Notebook, launchable by anyone using [Binder](https://mybinder.org/), describing the project and its steps. Python and Julia scripts of the machine learning tasks will also be provided.
