# Body Level Classification

## Table of Contents

- [Body Level Classification](#body-level-classification)
  - [Table of Contents](#table-of-contents)
  - [Problem and Dataset Description](#problem-and-dataset-description)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Dataset](#dataset)
    - [Univariate Analysis](#univariate-analysis)
      - [Target Variable Analysis](#target-variable-analysis)
      - [Features Analysis](#features-analysis)
        - [Numerical Features Distribution](#numerical-features-distribution)
        - [Categorical Features Distribution](#categorical-features-distribution)
    - [Multivariate Analysis](#multivariate-analysis)
      - [Correlation Matrix](#correlation-matrix)
      - [Numerical Feature Box Plot](#numerical-feature-box-plot)
      - [The relation between the features and the target variable](#the-relation-between-the-features-and-the-target-variable)
  - [Models Analysis](#models-analysis)
    - [Base Model](#base-model)
    - [Logistic Regression](#logistic-regression)
      - [Feature Importance Plot](#feature-importance-plot)
      - [Learning Curves Plot](#learning-curves-plot)
      - [Partial Dependence Plot](#partial-dependence-plot)
      - [Hyperparameter Tuning](#hyperparameter-tuning)
        - [Grid Search](#grid-search)
        - [Train-Validation Curve](#train-validation-curve)
      - [Bias-variance Analysis](#bias-variance-analysis)
      - [Decision Boundary Plot](#decision-boundary-plot)
    - [Support Vector Machines](#support-vector-machines)
      - [Feature Importance Plot](#feature-importance-plot-1)
      - [Learning Curves Plot](#learning-curves-plot-1)
      - [Partial Dependence Plot](#partial-dependence-plot-1)
      - [Hyperparameter Tuning](#hyperparameter-tuning-1)
        - [Grid Search](#grid-search-1)
        - [Train-Validation Curve](#train-validation-curve-1)
      - [Bias-variance Analysis](#bias-variance-analysis-1)
      - [Decision Boundary Plot](#decision-boundary-plot-1)
  - [Conclusion](#conclusion)
  - [Appendix](#appendix)
    - [Regularization](#regularization)
    - [Logistic Regression Parameters](#logistic-regression-parameters)
    - [The C parameter and the maximum-a-posterior Estimation](#the-c-parameter-and-the-maximum-a-posterior-estimation)
    - [Bias-variance Tradeoff](#bias-variance-tradeoff)

## Problem and Dataset Description

We are solving a classification problem for human body level based on some given attributes related to the physical, genetic and habitual conditions. The given attributes are both categorical and continuous. The human body level can be categorized into (4 levels/classes).
We are given 16 attributes and 1477 data samples, where classes are not evenly distributed. We are trying to build
models that can adapt to the class imbalance to achieve the best possible results.

## Exploratory Data Analysis

### Dataset

| Column             |  Data Type | Data Format |
| ------------------ |  --------- | ----------- |
| Gender             |  Nominal   | String      |
| Age                |  Ratio     | Float       |
| Height             |  Ratio     | Float       |
| Weight             |  Ratio     | Float       |
| H_Cal_Consump      |  Nominal   | String      |
| Veg_Consump        |  Interval  | Float       |
| Water_Consump      |  Interval  | Float       |
| Alcohol_Consump    |  Nominal   | String      |
| Smoking            |  Nominal   | String      |
| Meal_Count         |  Interval  | Float       |
| Food_Between_Meals |  Nominal   | String      |
| Fam_Hist           |  Nominal   | String      |
| H_Cal_Burn         |  Nominal   | String      |
| Phys_Act           |  Interval  | Float       |
| Time_E_Dev         |  Interval  | Float       |
| Transport          |  Nominal   | String      |
| Body_Level         |  Nominal   | String      |

### Univariate Analysis

The univariate analysis involves analyzing each variable individually. It looks at the range of values and the central tendency of the values. It describes the pattern of response to the variable. It represents each variable on its own.

#### Target Variable Analysis

We started by analyzing the distribution of the target variable and found an imbalance in our data set where one class dominated the others. This issue needs to be addressed as unequal representation of classes in our data set can lead to biased results when training machine learning algorithms. Techniques like oversampling the minority class, undersampling the majority class, or utilizing cost-sensitive sampling can be implemented to balance the data set.
<div style="text-align:center">
<img src="figures/dataset_analysis/body_level_distribution.png" alt="Target Variable Distribution" width="800" height="600">
</div>

#### Features Analysis

##### Numerical Features Distribution

- We can find that the `Height` follows a normal distribution.
- We can find that the `Age` feature is left skewed, we have more values for youth than the elders.
- We can find that the `Veg_Consump`, `Water_Consump`, `Meal_Count`, `Phys_Act`, and `Tine_E_Dev` are capped to be integers.
  
>> Data capping is a technique used in data science to limit the maximum value of a variable in a dataset. This is often done to avoid the influence of outliers, or extreme values, that can skew the results of statistical analysis or machine learning models. By setting a cap, any values above that limit are truncated to the maximum value, allowing the analysis to focus on the most representative data points within the dataset. .

<div style="text-align:center">
<img src="figures/dataset_analysis/numerical_features_distribution.png" alt="Numerical Features Distribution" width="800" height="600">
</div>

##### Categorical Features Distribution

- We can find that most of the features are unbalanced, this makes them somehow useless in solving our problem except for the `Gender` feature i.e. we may use this feature in our analysis.

<div style="text-align:center">
<img src="figures/dataset_analysis/categorical_feature.png" alt="Categorical Features Distribution" width="800" height="600">
</div>

### Multivariate Analysis

The multivariate analysis involves analyzing more than one variable to determine the relationship between them. It looks at the interactions between variables. It is used to identify patterns in the data set. It represents each variable in relation to all other variables.

#### Correlation Matrix

We can find that there' a correlation between the width and the height features which is expected.

<div style="text-align:center">
<img src="figures/dataset_analysis/correlation_matrix.png" alt="Correlation Matrix" width="800" height="600">
</div>

#### Numerical Feature Box Plot

We can notice that the `Weight` feature is an important and a significant feature. It carries a lot of predictive power and is highly correlated with the target variable. We can use it to build some models that can accurately predict the target variable.

<div style="text-align:center">
<img src="figures/dataset_analysis/numerical_features_boxplot.png" alt="Numerical Feature Box Plot" width="800" height="600">
</div>

#### The relation between the features and the target variable

We can notice that the `Weight` and `Height` are significant features that carry a lot of predictive power and are highly correlated with the target variable.. We can use them or a mix of them to build our models.

<div style="text-align:center">
<img src="figures/dataset_analysis/features_pairplot.png" alt="Features Pair Plot" width="1200" height="1000">
</div>

## Models Analysis

All the analysis is made on the **Width** and **Height** features, as we have shown that they are the most significant and important ones.

### Base Model

Before working on developing sophisticated machine learning models, we used some dummy models to as a baseline model to compare the performance of more sophisticated models. By comparing the performance of a complex model to that of a simple model, we can determine if the complex model is actually providing useful predictions or if it is overfitting the data. Dummy models also help identify if the problem has any inherent bias or if the dataset is imbalanced. Overall, starting with a dummy model is a good way to get a baseline understanding of the data and the problem before moving on to more complex models.

<!-- TODO: recalc based on the w and h only -->
| Strategy   | Description  | Cross-validation with 10 folds   |
|:---: |:---: |:---: |
| **most_frequent**  | The predict method always returns the most frequent class label in the observed y argument passed to fit. The predict_proba method returns the matching one-hot encoded vector.  | **accuracy**:  0.4589374732944025 **f1_macro**:  0.1572825648608684 **f1_micro**:  0.4589374732944025  |
| **stratified**  | The predict_proba method randomly samples one-hot vectors from a multinomial distribution parametrized by the empirical class prior probabilities. The predict method returns the class label which got probability one in the one-hot vector of predict_proba. Each sampled row of both methods is therefore independent and identically distributed.  | **accuracy**:  0.32177040307648486 **f1_macro**:  0.24690614458225352 **f1_micro**:  0.32177040307648486  |
| **uniform**  | Generates predictions uniformly at random from the list of unique classes observed in y, i.e. each class has equal probability  | **accuracy**:  0.23281583819968663 **f1_macro**:  0.21401372344564545 **f1_micro**:  0.23281583819968663  |

### Logistic Regression

Logistic regression is a powerful tool for predicting categorical outcomes. It is used in a wide variety of fields, including marketing, medicine, and finance. For example, logistic regression can be used to predict the likelihood that a customer will buy a product, the likelihood that a patient will develop a disease, or the likelihood that a company will go bankrupt.

- The advantages of using logistic regression
  - It is a simple and straightforward method to predict categorical outcomes.
  - It can be used to predict the probability of an outcome for any given combination of predictor values.
  - It is relatively easy to interpret the results of a logistic regression model.

- The disadvantages of using logistic regression
  - It can be sensitive to outliers in the data.
  - It can be difficult to interpret the results of a logistic regression model when there are multiple independent variables.
  - It can be computationally expensive to fit a logistic regression model with a large number of independent variables.

Overall, logistic regression is a powerful tool for predicting categorical outcomes. It is relatively easy to use and interpret, and it can be used in a wide variety of fields.

>> To learn more about the logistic regression parameters, check the [Logistic Regression Parameters](#logistic-regression-parameters) section in the Appendix.
>>
#### Feature Importance Plot

A feature importance plot shows the importance of each feature in the model. It can be used to identify the most important features and to understand the impact of each feature on the model's predictions.

<div style="text-align:center">
<img src="figures/Logistic Regression/feature_importance.png" alt="Logistic Regression feature_importance" width="500" height="400">
</div>

Based on the feature importance graph, it seems that `Weight` has a much higher importance compared to `Height` for the logistic regression model. In fact, `Weight` accounts for approximately 79% of the importance, while "Height" only accounts for 21%.

There are a few possible explanations for this. First, `Weight` is a more direct measure of body level than `Height`. Second,  `Weight` is more variable than `Height`, which means that it have a greater impact on the outcome variable.

#### Learning Curves Plot

show the training error (Ein) and validation error (Eval) as a function of the training set size

<div style="text-align:center">
<img src="figures/Logistic Regression/learning_curve.png" alt="Logistic Regression Learning Curves Plot" width="500" height="400">
</div>

<!-- TODO: Add the used params -->

The learning curves plot shows that the model is slightly overfitting the training data. The training error is lower than the validation error, which means that the model is not generalizing well to unseen data. This is a common problem with logistic regression models, and it can be addressed by using regularization techniques such as regularization as shown in the [Regularization](#regularization) section in the Appendix.

#### Partial Dependence Plot

A partial dependence plot shows the relationship between a feature and the model's predictions while holding all other features constant. It can be used to understand the impact of a single feature on the model's predictions i.e. PDPs show the average effect on predictions as the value of feature changes.

<div style="text-align:center">
<img src="figures/Logistic Regression/partial_dependencies.png" alt="Partial Dependence Plot" width="1000" height="400">
</div>

From the plots we can see the relationship between the `Weight` and `Height` features and the model's predictions. We can see that the relationship between the `Height` and the model's predictions is linear and positive, which means that the body level metric increases as the `Height` increases. We can also see that the relationship between the `Weight` and the model's predictions is non-linear and negative, which means that the body level metric decreases as the `Weight` increases.

#### Hyperparameter Tuning

This is a process of adjusting the parameters of a model to optimize its performance. It can be done using techniques like grid search, random search, or Bayesian optimization.

##### Grid Search

For

```python
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}
```

We found that the best parameters are:

```python
Best parameters found: {'C': 10, 'penalty': 'l1', 'solver': 'saga'}
```

With training score: `0.9710438585061565` and test score: `0.966636441509922`

Here are some heatmap visualizations of the grid search results:
<div style="display: flex; flex-direction: row; style="text-align:center"">
<img src="figures/Logistic Regression/hyper_param_heat_maps_C_solver.png" alt="hyper_param_heat_maps_C_solver" width="500" height="400">
<img src="figures/Logistic Regression/hyper_param_heat_maps_penalty_C.png" alt="hyper_param_heat_maps_penalty_C" width="500" height="400">
<img src="figures/Logistic Regression/hyper_param_heat_maps_penalty_solver.png" alt="hyper_param_heat_maps_penalty_solver" width="500" height="400">
</div>

##### Train-Validation Curve

Here are some Train-Validation Curves that we further used for the hyperparameter tuning process:

<div style="display: flex; flex-direction: row; style="text-align:center"">
<img src="figures/Logistic Regression/hyper_param_train_val_C_[0.001, 0.01, 0.1, 1, 10, 100].png" alt="hyper_param_train_val_C_[0.001, 0.01, 0.1, 1, 10, 100]" width="500" height="400">
<img src="figures/Logistic Regression/hyper_param_train_val_solver_['liblinear', 'saga'].png" alt="hyper_param_train_val_solver_['liblinear', 'saga']" width="500" height="400">
<img src="figures/Logistic Regression/hyper_param_train_val_penalty_['l1', 'l2'].png" alt="hyper_param_train_val_penalty_['l1', 'l2']" width="500" height="400">
</div>

#### Bias-variance Analysis

```yaml
bias:  0.09540938818565402
var:  0.024485126582278482
```

In our problem, we can see that the model has a low bias and a low variance, which means that it is well-fitted to the data and can generalize well to new, unseen data.
<!-- TODO: Add the used params -->

>> For more information on the Bias-variance tradeoff, read the [Bias-variance Tradeoff](#bias-variance-tradeoff) section in the appendix.

#### Decision Boundary Plot

This is plot shows the decision boundary of the model. It can help visualize how the model separates the classes in the dataset

<div style="text-align:center">
<img src="figures/Logistic Regression/decision_regions_for_Logistic Regression.png" alt="Decision Boundary Plot" width="1100" height="250">
</div>
<!-- TODO: Add the used params -->

<!-- TODO: Add a comment on this part -->

### Support Vector Machines

Support Vector Machines (SVMs) are a class of supervised learning algorithms used for classification and regression analysis. SVMs work by finding the hyperplane that best separates the data into different classes. The hyperplane that maximizes the margin between the two classes is chosen as the optimal decision boundary. In the case where the data is not linearly separable, SVMs use a kernel trick to map the data into a higher-dimensional space where the data can be linearly separated.

- Advantages of SVM:
  - Effective in high-dimensional spaces: SVMs can perform well in high-dimensional spaces, making them useful for solving complex problems with many features.
  - Robust against overfitting: SVMs are less prone to overfitting than other classification algorithms because they try to maximize the margin between classes, which helps prevent the model from being too closely fit to the training data.
  - Versatile: SVMs can be used for both linear and nonlinear classification and regression tasks, and different kernel functions can be used to handle different types of data.
  - Works well with small and large datasets: SVMs are computationally efficient and can work well with small and large datasets.

- Disadvantages of SVM:
  - Can be sensitive to the choice of kernel: SVM performance depends heavily on the choice of kernel, which can be challenging to choose correctly.
  - Requires careful preprocessing of data: SVMs are sensitive to the scale of the input features, so data preprocessing is required to standardize the features.
  - Computationally intensive: SVMs can be computationally intensive, particularly for large datasets and complex kernel functions.
  - Binary classification only: SVMs are designed for binary classification problems, which means they need to be modified for multi-class classification tasks.

In summary, SVMs are a powerful and flexible classification algorithm that can be used for a wide range of tasks. However, they require careful preprocessing of data, can be sensitive to the choice of kernel, and can be computationally intensive.

#### Feature Importance Plot

A feature importance plot shows the importance of each feature in the model. It can be used to identify the most important features and to understand the impact of each feature on the model's predictions.

<div style="text-align:center">
<img src="figures/SVM/feature_importance.png" alt="SVM feature_importance" width="500" height="400">
</div>

Based on the feature importance graph, it seems that `Weight` has a much higher importance compared to `Height` for the logistic regression model. In fact, `Weight` accounts for approximately 80% of the importance, while "Height" only accounts for 20%.

There are a few possible explanations for this. First, `Weight` is a more direct measure of body level than `Height`. Second,  `Weight` is more variable than `Height`, which means that it have a greater impact on the outcome variable.

#### Learning Curves Plot

show the training error (Ein) and validation error (Eval) as a function of the training set size

<div style="text-align:center">
<img src="figures/SVM/learning_curve.png" alt="SVM Learning Curves Plot" width="500" height="400">
</div>

<!-- TODO: Add the used params -->

The learning curves plot shows that the model is slightly overfitting the training data. The training error is lower than the validation error, which means that the model is not generalizing well to unseen data. This is a common problem with logistic regression models, and it can be addressed by using regularization techniques such as regularization as shown in the [Regularization](#regularization) section in the Appendix.

#### Partial Dependence Plot

A partial dependence plot shows the relationship between a feature and the model's predictions while holding all other features constant. It can be used to understand the impact of a single feature on the model's predictions i.e. PDPs show the average effect on predictions as the value of feature changes.

<div style="text-align:center">
<img src="figures/SVM/partial_dependencies.png" alt="Partial Dependence Plot" width="1000" height="400">
</div>

From the plots we can see the relationship between the `Weight` and `Height` features and the model's predictions. We can see that the relationship between the `Height` and the model's predictions is linear and positive, which means that the body level metric increases as the `Height` increases. We can also see that the relationship between the `Weight` and the model's predictions is non-linear and negative, which means that the body level metric decreases as the `Weight` increases.

#### Hyperparameter Tuning

This is a process of adjusting the parameters of a model to optimize its performance. It can be done using techniques like grid search, random search, or Bayesian optimization.

##### Grid Search

For

```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear']
}
```

We found that the best parameters are:

```python
Best parameters found: {'C': 10, 'penalty': 'l1', 'solver': 'saga'}
```

With training score: `0.973803108864967` and test score: `0.9729384514609276`

Here are some heatmap visualizations of the grid search results:
<div style="display: flex; flex-direction: row; style="text-align:center"">
<img src="figures/SVM/hyper_param_heat_maps_C_gamma.png" alt="hyper_param_heat_maps_C_gamma" width="500" height="400">
<img src="figures/SVM/hyper_param_heat_maps_C_kernel.png" alt="hyper_param_heat_maps_C_kernel" width="500" height="400">
<img src="figures/SVM/hyper_param_heat_maps_gamma_kernel.png" alt="hyper_param_heat_maps_gamma_kernel" width="500" height="400">
</div>

##### Train-Validation Curve

Here are some Train-Validation Curves that we further used for the hyperparameter tuning process:

<div style="display: flex; flex-direction: row; style="text-align:center"">
<img src="figures/SVM/hyper_param_train_val_C_[0.1, 1, 10, 100].png" alt="hyper_param_train_val_C_[0.1, 1, 10, 100]" width="500" height="400">
<img src="figures/SVM/hyper_param_train_val_gamma_[0.1, 1, 10, 100].png" alt="hyper_param_train_val_gamma_[0.1, 1, 10, 100]" width="500" height="400">
<img src="figures/SVM/hyper_param_train_val_kernel_['rbf', 'linear'].png" alt="hyper_param_train_val_kernel_['rbf', 'linear']" width="500" height="400">
</div>

#### Bias-variance Analysis

```yaml
bias:  0.09540938818565402
var:  0.024485126582278482
```
<!-- TODO: Add the used params -->

In our problem, we can see that the model has a low bias and a low variance, which means that it is well-fitted to the data and can generalize well to new, unseen data.

>> For more information on the Bias-variance tradeoff, read the [Bias-variance Tradeoff](#bias-variance-tradeoff) section in the appendix.

#### Decision Boundary Plot

This is plot shows the decision boundary of the model. It can help visualize how the model separates the classes in the dataset

<div style="text-align:center">
<img src="figures/SVM/decision_regions_for_SVM.png" alt="Decision Boundary Plot" width="1100" height="250">
</div>
<!-- TODO: add plot for linear and rbf kernels -->
<!-- TODO: Add the used params -->

<!-- TODO: Add a comment on this part -->

## Conclusion

After some data exploration, we noticed that there is an obvious relation between the target variable and the weight, and height as shown here:

<div style="text-align:center">
<img src="figures/equation/w_H_target.png" alt="The relation between the Body Level and the weight, height" width="500" height="400">
</div>

We tried to find the best formula that can represent this relation and decided to use the following formula and got some fascinating results:

$$weightOverHeightSquared = \frac{weight}{height^2}$$

<div style="display: flex; flex-direction: row; style="text-align:center"">
<img src="figures/equation/WH_target.png" alt="Body Level vs Weight over Height Squared" width="500" height="400" style="width:50%; margin-right: 10px;">

<img src="figures/equation/WH_target_catplot.png" alt="Body Level vs Weight over Height Squared" width="500" height="400" style="width:50%; margin-left: 10px;">
</div>

**ThresholdClassifier:**

We constructed a custom classifier called `ThresholdClassifier`, that uses the aforementioned formula to predict the body level based on some thresholds.

**Threshold Selection:**

We tried using different thresholds to predict the body level like the following:

- Using the **min value** of the weightOverHeightSquared as a threshold.
- Using the **25th percentile** of the weightOverHeightSquared as a threshold.
- Using the **mean value** of the weightOverHeightSquared as a threshold.

We found that using the min value of the weightOverHeightSquared as a threshold gives the best results.

We undergone more research and found that that this metric is called BMI(Body Mass Index). Where: $$BMI = \frac{weight}{height^2}$$

|  Body Level  | Threshold  |                                                      Health Implication                                                      |
|:-----------: |:---------: |:---------------------------------------------------------------------------------------------------------------------------: |
| Underweight  |   <18.5    |                     Increased risk of health problems such as osteoporosis, heart disease, and diabetes                      |
|    Normal    | 18.5-24.9  |                                                     Healthy body weight                                                      |
|  Overweight  |  25-29.9   |       Increased risk of health problems such as heart disease, stroke, type 2 diabetes, and certain types of cancer          |
|   Obesity    |    30<     | Significantly increased risk of health problems such as heart disease, stroke, type 2 diabetes, and certain types of cancer  |

<!-- TODO: don't use cross validation, use the whole data as test -->
Henceforth, we decided to use the BMI thresholds to predict the body level and got the following results after running a cross-validation with 10 folds:
|  Metric  |  Value  |
|:-------: |:---------: |
|  accuracy   |  0.9864549209514315  |
|  f1_macro   |  0.982450234501931  |
|  f1_micro   |  0.9864549209514315  |

## Appendix

### Regularization

L2 regularization, also known as ridge regression, is a common technique used in machine learning to prevent overfitting and improve the generalization performance of a model. It achieves this by adding a penalty term to the objective function of the model.

The penalty term is proportional to the square of the L2 norm of the model's weight vector. The L2 norm of a vector is the square root of the sum of the squares of its components. In the case of the weight vector, this is equivalent to summing the squares of the individual weights. The penalty term is multiplied by a hyperparameter, lambda, which determines the strength of the regularization.

The L2 regularization objective function can be expressed as:

$$
J(\boldsymbol{\omega}) = \frac{1}{2\sigma^2}\left\lVert X\boldsymbol{\omega} - y\right\rVert_2^2 -\frac{\lambda}{2}\left\lVert \boldsymbol{\omega}\right\rVert_2^2
$$

where:

- $\boldsymbol{\omega}$ is the weight vector.
- $X$ is the input matrix.
- $y$ is the output vector.
- $\sigma^2$ is the variance.
- $\lambda$ is the regularization parameter.

The first term in this equation is the mean squared error between the predicted output and the true output, and the second term is the L2 regularization penalty term.

When training a model with L2 regularization, the objective is to find the weight vector that minimizes this objective function. By adding the L2 regularization term, the model is encouraged to find a weight vector that not only fits the training data well but also has smaller individual weights, which can help to reduce overfitting.

In practice, the value of the regularization parameter, lambda, is typically chosen through cross-validation. Larger values of lambda result in stronger regularization and smaller weights, while smaller values of lambda result in weaker regularization and larger weights. The appropriate value of lambda depends on the specific problem and data set, and finding the optimal value requires experimentation and tuning.

### Logistic Regression Parameters

In the scikit-learn implementation of logistic regression, the following parameters can be specified:

- `penalty`: This parameter controls the regularization penalty applied to the coefficients. Regularization helps prevent overfitting by reducing the magnitude of the coefficients. The two options for `penalty` are `'l1'` and `'l2'`. L1 regularization adds the absolute values of the coefficients to the objective function, while L2 regularization adds the squared values. The default value is `'l2'`.
- `C`: This is the inverse of the regularization strength and controls the trade-off between fitting the training data well and having small coefficients. Larger values of `C` result in less regularization and a better fit to the training data. The default value is `1.0`.
- `fit_intercept`: This parameter controls whether or not to include an intercept term in the model. An intercept term allows the model to capture the bias or constant term in the data. The default value is `True`.
- `solver`: This parameter specifies the algorithm to use for optimization. The options are `'lbfgs'`, `'liblinear'`, `'newton-cg'`, `'sag'`, and `'saga'`. The default value is `'lbfgs'`.
- `max_iter`: This parameter controls the maximum number of iterations for the solver to converge. The default value is `100`.

Now let's dive into the math behind logistic regression. Given a set of input variables `X` and a binary target variable `y`, logistic regression models the probability of `y` being 1 as follows:

$P(y=1|X) = \sigma(z) = \frac{1}{1 + \exp(-z)}$

where z is a linear combination of the input variables and their corresponding coefficients:

$z = b_0 + b_1 *x_1 + b_2* x_2 + ... + b_n * x_n$

The coefficients $(b_1, b_2, ..., b_n)$ are learned from the training data using a maximum likelihood estimation approach. The objective is to maximize the likelihood of observing the training data given the parameters of the logistic regression model.

To prevent overfitting, regularization is added to the objective function. L1 regularization adds the absolute values of the coefficients to the objective function, while L2 regularization adds the squared values. The strength of regularization is controlled by the `C` parameter, which is the inverse of the regularization strength.
>> Make sure to check [The C parameter and the maximum-a-posterior Estimation](#the-c-parameter-and-the-maximum-a-posterior-estimation) section.

The optimization problem is solved using an iterative algorithm that updates the coefficients until convergence. The specific algorithm used is controlled by the `solver` parameter. For example, the `'lbfgs'` solver uses the Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm, while the `'liblinear'` solver uses a coordinate descent algorithm.

In summary, logistic regression is a powerful algorithm for binary classification tasks that models the probability of a binary response variable as a function of one or more predictor variables. By controlling the regularization penalty and optimization algorithm, logistic regression can be tuned to achieve better performance on a wide range of problems.

### The C parameter and the maximum-a-posterior Estimation

Using regularization is equivalent to using a maximum-a-posterior Estimation with a prior distribution (We assume conjugate prior to the likelihood to obtain a closed-form representation of posterior.)

The MAP criterion is derived from Bayes Rule, i.e.

```math
\begin{equation}
P(A \vert B) = \frac{P(B \vert A)P(A)}{P(B)} 
\end{equation}
```

If $B$ is chosen to be our data $X$ and $A$ is chosen to be the parameters that you'd want to estimate, call it $w$, we will get

```math
\begin{equation}
\underbrace{P(w \vert {X})}_{\text{Posterior}} = 
\frac{1}{\underbrace{P({X})}_{\text{Normalization}}}
\overbrace{P({X} \vert w)}^{\text{Likelihood}}\overbrace{P(w)}^{\text{Prior}}  \tag{0}
\end{equation}
```

And by introducing $y$:

$$
p(\boldsymbol{\omega} | X, y) = \frac{p(y | X, \boldsymbol{\omega})P(\boldsymbol{\omega})}{p(y | X)}
$$

For mathematical convince, we can write is as follows:
$$\boldsymbol{\hat{\omega}}_{MAP} = \arg\max_{\boldsymbol{\omega}} p(y | X, {\omega})P({\omega})$$

The above equation is the Maximum A Posteriori (MAP) estimation for the parameter vector $\boldsymbol{\omega}$, given the data matrix $X$ and the output vector $y$. It aims to find the values of $\boldsymbol{\omega}$ that maximize the posterior probability $p(\boldsymbol{\omega} | X, y)$.

Taking the logarithm of both sides of the equation above, we have:

$$
\ln p(\boldsymbol{\omega} | X, y) = \ln p(y | X, \boldsymbol{\omega}) + \ln P(\boldsymbol{\omega}) - \ln p(y | X)\\
$$

Since the term $-\ln p(y | X)$ is constant with respect to $\boldsymbol{\omega}$, maximizing $\ln p(\boldsymbol{\omega} | X, y)$ is equivalent to maximizing $\ln p(y | X, \boldsymbol{\omega}) + \ln P(\boldsymbol{\omega})$. Using the logarithmic identities $\ln a + \ln b = \ln ab$ and $\ln a^n = n\ln a$, we can further simplify the expression as:

$$
\ln p(y | X, \boldsymbol{\omega}) + \ln P(\boldsymbol{\omega}) = \sum_{i=1}^n \ln p(y_i | \boldsymbol{x}_i, \boldsymbol{\omega}) + \ln P(\boldsymbol{\omega})\\
= \sum_{i=1}^n \ln p(y_i | \boldsymbol{x}_i, \boldsymbol{\omega}) + \ln P(\boldsymbol{\omega})$$

We can take the $-$ sign outside and convert our problem into a minimization one as follows:
$$\boldsymbol{\hat{\omega}}_{MAP} = \arg\min_{\boldsymbol{\omega}} p(y | X, {\omega})P({\omega})$$

$$
= \sum_{i=1}^n \ln \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i - \boldsymbol{x}_i^T\boldsymbol{\omega})^2}{2\sigma^2}\right) + \ln P(\boldsymbol{\omega})$$
We can remove some of the constants as they won't affect our result.

$$
= \frac{1}{2\sigma^2}\left\lVert X\boldsymbol{\omega} - y\right\rVert_2^2 +  \ln P(\boldsymbol{\omega})\\
$$

where $\boldsymbol{x}_i$ is the $i$-th row of $X$, $\sigma^2$ is the variance of the noise in the data, and $\left\lVert X\boldsymbol{\omega} -y \right\rVert_2^2$ is the squared Euclidean distance between the predicted outputs $X\boldsymbol{\omega}$ and the true outputs $y$.

Now, we can expand the second term of the given equation to the L2 regularization form as follows:

$$
\ln P(\boldsymbol{\omega}) = \ln \exp(-\frac{\lambda}{2}\left\lVert \boldsymbol{\omega}\right\rVert_2^2) \\
= -\frac{\lambda}{2}\left\lVert \boldsymbol{\omega}\right\rVert_2^2
$$

where $\lambda$ is a hyperparameter that controls the strength of regularization and we used the fact that $\ln \exp(x) = x$.

Substituting this into the given equation, we get:

$$
= \frac{1}{2\sigma^2}\left\lVert X\boldsymbol{\omega} - y\right\rVert_2^2 -\frac{\lambda}{2}\left\lVert \boldsymbol{\omega}\right\rVert_2^2
$$

This is the same as the cost function with L2 regularization, where the regularization parameter $\lambda$ controls the strength of regularization.

The L2 regularization term penalizes the magnitude of the coefficients of the model by adding a penalty term to the cost function. This penalty term is proportional to the square of the L2 norm of the coefficient vector. Mathematically, the L2 regularization term can be expressed as:

### Bias-variance Tradeoff

Bias refers to the error that is introduced by approximating a real-world problem with a simplified model. A high bias model is one that is too simplistic and cannot capture the underlying patterns in the data. This often results in an underfit model that has high training error and poor performance on both the training and testing datasets. On the other hand, a low bias model is one that is complex enough to capture the underlying patterns in the data. This often results in a well-fitted model that has low training error and good performance on both the training and testing datasets.

Variance, on the other hand, refers to the amount by which the prediction of the model would change if we trained it on a different dataset. A high variance model is one that is too sensitive to the training data and cannot generalize well to new, unseen data. This often results in an overfit model that has low training error but poor performance on the testing dataset. On the other hand, a low variance model is one that is less sensitive to the training data and can generalize well to new, unseen data. This often results in a well-fitted model that has low training error and good performance on the testing dataset.

In general, the goal is to find a model with both low bias and low variance. This is often achieved through:

- Feature selection: Selecting a subset of features that are most relevant to the target variable. That's why we chose the `Weight` and `Height` features.
- Regularization: This involves adding a penalty to the loss function, which helps to reduce the complexity of the model as discussed in the previous sections.
- Ensemble learning: This involves combining the predictions of multiple models, which helps to reduce the variance of the predictions.

It is important to note that the trade-off between bias and variance depends on the specific problem and dataset at hand. In some cases, a more complex model may be necessary to capture the underlying patterns in the data, even if it comes at the cost of higher variance. In other cases, a simpler model may be sufficient, even if it comes at the cost of higher bias.
