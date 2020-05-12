# Content

- <a href="#captumigshapbaseline-anchor">captumig-shap-baselines</a> [(notebook)](captumIg_shap_baselines.ipynb)


# <a name="captumigshapbaseline-anchor">Captum Integrated Gradients and SHAP for a PyTorch MPG prediction model</a>
---

This notebook contains an example of two feature attribution methods applied to a PyTorch model predicting fuel efficiency for the [Auto MPG Data Set](http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data). 

We will use the following methods:
- [Integrated Gradients from the Captum package](https://captum.ai/api/integrated_gradients.html)
- custom toy implementation of the [SHAP algorithm (Shapley values)](https://en.wikipedia.org/wiki/Shapley_value)

Attribution methods are applied per sample. As a result, each  feature is assigned a value reflecting its contribution to the model's output or, more precisely, to the difference between model's output for the sample and the *expected value*. 

Both methods used in this notebook require setting a baseline, i.e.: a vector of values that will be used, for each feature, in place of a missing value. The baseline vector serves as a set of reference values that can be thought of as *neutral* and that are used to represent a missing value whenever a method requires it. We will calculate the *expected value* as model's output for a selected baseline. 

All attributions together account for the difference between the model's prediction for a sample and the expected value of the model's output for a selected baseline. 


In the examples below we will consider various baselines and see how they influence assigning importance to features.
We will see that, for each sample, attributions sum up to the difference between model's output for the sample and the *expected value* (model's output for the baseline used to compute attributions).

## Attributions explain prediction
<img src="imgs/explain-diff-ig.png" width="800px" style="max-width:100%"> | 
------------ | 
Attributions sum up to the difference between model's output and the expected value (model's output for the baseline vector).

## Features and attributions
<img src="imgs/attr-features-1.png" width="800px" style="max-width:100%"> | 
------------ | 
The diagrams show how high and low values of features are distributed across the range of attributions assigned by IG and SHAP for various baselines. For some features, high values of the feature (in red) correlate with high values of attributions (x-axis), for some they gather in the lower range or there is no clear correlation.  |

## Impact of features
<img src="imgs/features-sum-12.png" width="800px" style="max-width:100%"> | 
------------ | 
Accumulated feature importance varies more between baselines than it does between attribution methods. One intuitive explanation is that since both methods use a baseline to stand for a missing value, features that have close to monotonic relationship to the target will be more consistently attributed a higher absolute impact when replaced by a zero.

