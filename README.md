# Captum Integrated Gradients and SHAP for a PyTorch MPG prediction model

This notebook contains an example of two feature attribution methods applied to a PyTorch model predicting fuel efficiency for the [Auto MPG Data Set](http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data). 

We will use the following methods:
- [Integrated Gradients from the Captum package](https://captum.ai/api/integrated_gradients.html)
- custom toy implementation of the [SHAP algorithm (Shapley values)](https://en.wikipedia.org/wiki/Shapley_value)

Attribution methods are applied per sample. As a result, each  feature is assigned a value reflecting its contribution to the model's output or, more precisely, to the difference between model's output for the sample and the *expected value*. 

Both methods used in this notebook require setting a baseline, i.e.: a vector of values that will be used, for each feature, in place of a missing value. The baseline vector serves as a set of reference values that can be thought of as *neutral* and that are used to represent a missing value whenever a method requires it. We will calculate the *expected value* as model's output for a selected baseline. 

All attributions together account for the difference between the model's prediction for a sample and the expected value of the model's output for a selected baseline. 


In the examples below we will consider various baselines and see how they influence assigning importance to features.
We will see that, for each sample, attributions sum up to the difference between model's output for the sample and the *expected value* (model's output for the baseline used to compute attributions).
