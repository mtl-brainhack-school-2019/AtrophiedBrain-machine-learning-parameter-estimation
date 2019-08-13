# machine-learning-parameter-estimation
Repository for AtrophiedBrain, inherited and inspired by Sylvie Rheault's BrainHack School 2018 project https://github.com/mtl-brainhack-school-2018/rheauls

## Background
The general objectives of this BrainHack project is to use machine learning techniques to estimate the parameters of mathematical models.

## Specific learning objectives and deliverables
During the BrainHack school, I hope to achieve to following goals:

### Regression and machine learning

1) Starting from a fictive equation and data simulated with the equation, test machine learning tools to find the weight of parameters for the linear model.  For example:

   f(t,u,v,w,x,y,z) = 2.1*t + 3.6*u + 2.8*v + 5*w + 1.4*x - 2.5*y + 4*z
   
   f()=cognition
   
   t=systolic blood pressure/140 | u=diastolic blood pressure/90 | v=glycosylated Hb/7 | w=age/70 | x=LDL/2 | y=years of education/12 | z=VS/20 |   
   
2) Introduce noise to the simulated data.

3) Redo data simulation using a non-linear equation.

4) Use machine learning to solve this regression task

### ODE parameter estimation and machine learning

5) Define a parameterized, ODE-based time series problem

6) Generate noisy training data from the ODEs

6) Use machine learning techniques to recover the original parameter estimates from the noisy data.
