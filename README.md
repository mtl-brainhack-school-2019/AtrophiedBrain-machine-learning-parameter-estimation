# Machine Learning Parameter Estimation
Repository for AtrophiedBrain, inherited and inspired by Sylvie Rheault's BrainHack School 2018 project https://github.com/mtl-brainhack-school-2018/rheauls

## Summary
The structures of the brain change naturally with aging and diseases modify the brain in their own way. Increased knowledge of the location and timing of the changes caused by disease could teach us about how diseases progress. This project used machine learning to model how the brain naturally changes.


## Project definition
Unfortunately, we do not fully understand how the brain normally changes. A model of how the structures of the brain change in normal aging could serve as a baseline for comparison with models of specific diseases. Neural networks are proving to be useful for many scientific tasks, can they be useful for mathematical modeling?


Ordinary differential equations (ODEs) are commonly used to model biological systems. A [recent paper](https://arxiv.org/abs/1806.07366) introduced the concept of Neural Ordinary Differential Equations that reportedly allow ODE models to be trained using deep learning tools. This project explores the use of Neural ODEs for estimating the parameters of an ODE-based model of the change of grey matter in the brain.


A simple single-parameter model of Cortical Thickness was chosen as the model to be studied using Neural ODEs. This single parameter is effectively the rate of atrophy or growth of each region of the brain. Click HERE for an interactive demo of this model.


## Learning experience
This project primarily used [Python](https://www.python.org/), [PyTorch](https://pytorch.org/) (Python’s deep learning framework), and [Jupyter Notebooks](https://jupyter.org/) stored in a public GitHub repository. [Dash by Plotly](https://plot.ly/dash/) was used to make interactive visualizations. [Julia](https://julialang.org/) and Julia’s deep learning library [Flux.jl]( https://fluxml.ai/) were also used, as their Neural ODE library [DiffEqFlux]( https://github.com/JuliaDiffEq/DiffEqFlux.jl) is under heavy development (click [here]( https://www.youtube.com/watch?v=5ZgEp36E71Y&amp=&index=37&amp=&t=0s) for an introduction).


## Project deliverables
1. Jupyter Notebook that defines a linear regression function, generates data from this function, adds noise to the data, defines a model to recover the model parameters from the linear model, splits the data into train/test sets, trains the model, and presents the results.
2. Jupyter Notebook that defines an ODE-based model and performs identifiability analysis on the model parameters.
3. Jupyter Notebook that uses a validated ODE-based model to generate data, adds noise to the data, defines a model to recover the model parameters, splits the data into train/test sets, trains the model, recovers the model parameters, and presents the results.

## Dataset
All data for this project are simulated. 

### Linear regression
For the linear regression task, the following function is being used:
   f(t,u,v,w,x,y,z) = 2.1*t + 3.6*u + 2.8*v + 5*w + 1.4*x - 2.5*y + 4*z
   
   f()=cognition
   
   t=systolic blood pressure/140 | u=diastolic blood pressure/90 | v=glycosylated Hb/7 | w=age/70 | x=LDL/2 | y=years of education/12 | z=sedimentation rate/20 |   
   
Example data:
```
   bp_sys  bp_dia     hba1c   age       ldl   edu  sed_rate
0   147.0    87.0  6.936874  57.0  2.795019  14.0      32.0
1   127.0    54.0  7.876324  76.0  5.674918  14.0      69.0
2    93.0    76.0  9.657790  70.0  4.546000   9.0      32.0
3   132.0    63.0  3.742432  74.0  1.960318  12.0      18.0
4   175.0    61.0  6.932590  74.0  8.334432   5.0      25.0
```
   
![Linear Regression](https://raw.githubusercontent.com/mtl-brainhack-school-2019/AtrophiedBrain-machine-learning-parameter-estimation/master/figures/linear_regression.png)
   
### ODE-based model
dx/dt = a*x + b*(y + z)

dy/dt = a*y + b*(x + z)

dz/dt = a*z + b*(x + y)

Parameters: -1  	&#8804; a, b  	&#8804; 1 

## Tools/Languages used
- <img src="https://logodix.com/logo/729548.png" width=64>
- <img src="https://github.com/favicon.ico" width=64> GitHub
- <img src="https://res-2.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_256,w_256,f_auto,q_auto:eco/v1463481639/zkwcls2ljise1px6w3l6.png" width=64> Notebook
- <img src="https://static.nvidiagrid.net/ngc/containers/pytorch-logo-light.png" width=64> PyTorch
- Pydentify, which performs identifiability analysis by performing a Parameter Profile Analysis using COPASI
https://www.youtube.com/watch?v=F8l00uU141o

## Machine learning model architecture
Coming soon!

## To-Do
- [X] Create initial script for linear regression task
- [X] Add example of data (perhaps view of Panda DataFrame head) to Dataset section of this project description
- [ ] Add figures to illustrate the project and data
- [ ] Finish linear regression script
- [ ] Add figure of solved regression task
- [ ] Convert linear regression script into a Jupyter notebook
- [X] Find Python library for parameter identifiability analysis
- [X] Define initial ODE system
- [ ] Write script to generate noisy data from ODE model
- [ ] Write script to perform ODE parameter identifiability analysis
- [ ] Convert above two scripts into a Jupyter Notebook for model validation
- [ ] Create a Jupyter Notebook for using machine learning to estimate the ODE-based model's parameters
