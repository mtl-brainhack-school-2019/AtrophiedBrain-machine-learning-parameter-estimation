# Machine learning parameter estimation
Repository for AtrophiedBrain, inherited and inspired by Sylvie Rheault's BrainHack School 2018 project https://github.com/mtl-brainhack-school-2018/rheauls

## Background
The general objectives of this BrainHack project are to use machine learning techniques to estimate the parameters of mathematical models.

## Learning aims
1. Linear regression using machine learning
2. Validation of mathematical models
3. Parameter identifiability analysis
4. Learn more about Python data science ecosystem
5. What are neural ODEs?

## Project aims
1. Generate noisy data from an equation and to recover the coefficients from the noisy data using machine learning.
2. Validate ODE based model, focusing on indentfiability analysis of the model parameters.
alidate the model, investigating the identifiability of the parameters using 

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
   
![Linear Regression](https://raw.githubusercontent.com/mtl-brainhack-school-2019/AtrophiedBrain-machine-learning-parameter-estimation/master/figures/linear_regression.png)
   
### ODE-based model
dx/dt = a*x + b*(y + z)

dy/dt = a*y + b*(x + z)

dz/dt = a*z + b*(x + y)

Parameters: -1  	&#8804; a, b  	&#8804; 1 

## Tools/Languages used
- Python
- Jupyter Notebook
- PyTorch
- Pydentify, which performs identifiability analysis by performing a Parameter Profile Analysis using COPASI (https://www.youtube.com/watch?v=F8l00uU141o)

## Machine learning model architecture
ToDo!

## To-Do
- [X] Create initial script for linear regression task
- [ ] Add example of data (perhaps view of Panda DataFrame head) to Dataset section of this project description
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
