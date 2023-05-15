# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.evaluate import bias_variance_decomp  # pip install mlxtend
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import (GridSearchCV, learning_curve,
                                     validation_curve)
from sklearn.preprocessing import StandardScaler

sns.set_style(
    style='darkgrid',
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
###############################################
# Reading the data


def _getFeatureLists():
    numerical_features = ['Weight', 'Height', 'Age',
                          'Meal_Count', 'Phys_Act', 'Water_Consump']
    categorical_features = ['Smoking', 'Alcohol_Consump', 'Transport']
    target = ['Body_Level']
    return numerical_features, categorical_features, target


def getData(path='../data/train_data.csv', scaleNumericalFeatures=False):
    # read the data
    df = pd.read_csv(path)
    # read the feature list
    numerical_features, categorical_features, target = _getFeatureLists()
    # Construct the data frame
    df = df[numerical_features + categorical_features + target]
    # encode categorical features
    df = pd.get_dummies(df, columns=categorical_features)
    # scale the numerical features
    if scaleNumericalFeatures:
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
    # prepare the data
    X = df.drop('Body_Level', axis=1)
    y = df['Body_Level']
    y = y.map({'Body Level 1': 0, 'Body Level 2': 1,
              'Body Level 3': 2, 'Body Level 4': 3})
    return X, y

# Feature Importance


def getFeatureImportance(features, importance):
    # Add the feature importances into a dataframe
    feature_importance = pd.DataFrame(
        {'feature': features, 'importance': importance})
    feature_importance.sort_values('importance', ascending=False, inplace=True)
    return feature_importance


def getFeatureImportancePlot(feature_importance, save=True, modelname='model'):
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    if save:
        plt.savefig(f'../figures/{modelname}/feature_importance.png', dpi=300, bbox_inches='tight')
    return plt

# Learning Curves


def getLearningCurvePlot(estimator, X, y, cv=5, scoring='f1_macro', modelname='model', save=True):
    # It uses cross-validation with cv folds
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,
                                                            train_sizes=np.linspace(
                                                                .1, 1.0, 5),
                                                            scoring='f1_macro', shuffle=True, random_state=42)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.title(f'Learning Curves for {modelname}')
    plt.xlabel('Training Set Size')
    plt.ylabel(f'1 - {scoring}')
    plt.ylim(0.0, 1.1)
    plt.grid()

    plt.fill_between(train_sizes, 1-(train_scores_mean - train_scores_std),
                     1-(train_scores_mean + train_scores_std), alpha=0.1, color='r')
    plt.fill_between(train_sizes, 1-(test_scores_mean - test_scores_std),
                     1-(test_scores_mean + test_scores_std), alpha=0.1, color='g')

    plt.plot(train_sizes, 1-train_scores_mean, 'o-', color='r', label='Ein')
    plt.plot(train_sizes, 1-test_scores_mean, 'o-', color='g', label='Eval')

    plt.legend(loc='best')
    if save:
        plt.savefig(f'../figures/{modelname}/learning_curve.png', dpi=300, bbox_inches='tight')
    return plt

# Partial Dependence Plot


def getPartialDependenciesPlot(estimator, X, modelname='model', save=True):
    fig, ax = plt.subplots(figsize=(20, 20), )
    target_class = 0  # specify the target class
    PartialDependenceDisplay.from_estimator(
        estimator, X, X.columns, ax=ax, target=target_class)

    # Increase spacing between subplots
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    fig.suptitle(f'Partial Dependence Plots for {modelname}')
    fig.tight_layout()
    if save:
        plt.savefig(f'../figures/{modelname}/partial_dependencies.png', dpi=300, bbox_inches='tight')
    return plt

# Grid Search


def getGridSearch(estimator, param_grid, X, y, scoring, cv=10):
    # Grid Search
    grid_search = GridSearchCV(
        estimator, param_grid, cv=cv, scoring=scoring, return_train_score=True)
    grid_search.fit(X, y)
    return grid_search


def plotHyperParamHeatMaps(param_grid, grid_search, modelname='model', save=True):
    # Create dataframe of validation accuracy for each hyperparameter combination
    results = pd.DataFrame(grid_search.cv_results_)[
        ['params', 'mean_test_score']]
    for param in param_grid:
        results[param] = results['params'].apply(lambda x: x[param])
    results.drop(columns=['params'], inplace=True)

    # Loop through all combinations of hyperparameters and plot heatmap of validation accuracy
    for i, param1 in enumerate(param_grid.keys()):
        for j, param2 in enumerate(list(param_grid.keys())[i+1:]):
            heatmap_data = results.pivot_table(
                index=param1, columns=param2, values='mean_test_score')
            heatmap_data.index = heatmap_data.index.astype(str)

            # Plot heatmap of validation accuracy for each hyperparameter combination
            sns.heatmap(heatmap_data, annot=True, cmap='coolwarm')
            plt.title(f"Validation f1_macro for {param1} and {param2} using {modelname}")
            if save:
                plt.savefig(f'../figures/{modelname}/hyper_param_heat_maps_{param1}_{param2}.png', dpi=300, bbox_inches='tight')
            plt.show()


# Train-Validation Curve
def plotHyperParamTrainValidationCurve(estimator, param_grid, X, y, cv=10, scoring='f1_macro', modelname='model', save=True):

    # iterate over the parameters and get the key and value pairs
    for param, value in param_grid.items():
        # Calculate training and validation scores for different values of max_depth
        train_scores, valid_scores = validation_curve(estimator, X, y,
                                                      param_name=param, param_range=value,
                                                      cv=cv, scoring=scoring)

        # Calculate the mean and standard deviation of the training and validation scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)
        valid_std = np.std(valid_scores, axis=1)

        # Plot the bias-variance tradeoff
        plt.plot(value, train_mean, label='Training score', color='blue')
        plt.fill_between(value, train_mean - train_std,
                         train_mean + train_std, alpha=0.2, color='blue')
        plt.plot(value, valid_mean, label='Cross-validation score', color='red')
        plt.fill_between(value, valid_mean - valid_std,
                         valid_mean + valid_std, alpha=0.2, color='red')
        plt.legend()
        plt.xlabel(param)
        plt.ylabel(scoring)
        plt.title(f'Bias-Variance Tradeoff for {param} using {modelname}')
        if save:
            plt.savefig(f'../figures/{modelname}/hyper_param_train_val_{param}_{value}.png', dpi=300, bbox_inches='tight')
        plt.show()


# Bias-Variance Analysis
def getBiasVariance(estimator, X_train, y_train, X_test, y_test):
    # convert X, y, X_test, y_test to numpy arrays
    XX = X_train.to_numpy()
    yy = y_train.to_numpy()
    XX_test = X_test.to_numpy()
    yy_test = y_test.to_numpy()

    # perform the bias-variance analysis
    mse, bias, var = bias_variance_decomp(
        estimator, X_train=XX, y_train=yy, X_test=XX_test, y_test=yy_test, loss='mse', random_seed=42)
    return mse, bias, var

def _getDecisionContours(estimator, f1, f2, y, ax):
    # plot the decision boundary
    x_min, x_max = f1.min() - 0.5, f1.max() + 0.5
    y_min, y_max = f2.min() - 0.1, f2.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))

    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    contours = ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(f1, f2, c=y, s=20, edgecolor='k')

    return contours
def getDecisionRegions(estimator, X, f1, f2, y, C=[0.001, 0.1, 10], modelname='model', save=True):
    # reset style to default
    # plt.style.use('default')

    # create a decision boundary plot at 3 different C values
    fig, axes = plt.subplots(1, 3, figsize=(25, 6))
    formatter = plt.FuncFormatter(lambda val, loc: ['Body Level 1', 'Body Level 2', 'Body Level 3', 'Body Level 4'][val])
    X = X[['Weight', 'Height']]
    for i, c in enumerate(C):
        estimator.set_params(C=c)
        estimator.fit(X, y)
        contours = _getDecisionContours(estimator, f1, f2, y, axes[i])
        axes[i].set_title('C = ' + str(c), size=15)
        plt.colorbar(contours, ticks =[0, 1, 2, 3],format=formatter)
    plt.title(f'Decision Regions using {modelname}')
    if save:
        plt.savefig(f'../figures/{modelname}/decision_regions_for_{modelname}.png', dpi=300, bbox_inches='tight')
    return plt
