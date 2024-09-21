# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary

This dataset contains data about a Portuguese banking institution's marketing campaigns (phone calls). It has 32950 items, 20 features about `customers` and information of the `marketing campaigns`.
We seek to predict whether customers will subscribe to a bank term deposit.

The solution will build and optimize an Azure ML pipeline using the model `Logistic Regression` with HyperDrive and AutoML to test on multiple models to find the best-performing model (higher accuracy).

## Scikit-learn Pipeline

The pipeline architecture includes 3 main steps:
1. Getting, cleaning and transforming data from https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv 
2. Applying Scikit-learn Logistic Regression algorithm with adjusting hyperparameters Regularization Strength `C` and Max iterations `max_iter`
3. Using HyperDrive to optimize hyperparameters call train.py to figure out the best run and save it.

**What are the benefits of the parameter sampler you chose?**

The parameter sampler I chose is:

```
ps = RandomParameterSampling({
    '--C': choice(0.01, 0.1, 0.3, 0.6, 0.7, 1.0),
    '--max_iter': choice(range(10,110,10))
    }
)
```

I chose a discrete set of values ​​for the parameter sampler because that is faster and supports early termination of low-performing activities.

The other techniques (Grid and Bayesian) are chosen if you have the budget for an exhaustive search. Additionally, Bayesian does not allow the use of early termination.

**What are the benefits of the early stopping policy you chose?**

I used the policy BanditPolicy with the tolerance threshold = 0.1 (A run is stopped if its primary metric is worse than the best run's metric by a factor of 10%), evaluation interval = 2 (the evaluation is done every 2 iterations), delay evaluation = 5 (delays the first evaluation until after 5 iterations).

Using BanditPolicy with these parameters improves resource management efficiency, accelerates experimentation, and improves overall hyperparameter tuning efficacy.

## AutoML

Azure AutoML tested multiple models on the training dataset, tuning hyperparameters to find the optimal model with higher accuracy.
The best result model was 'MaxAbsScaler LightGBM' with accuracy was 0.91555387. The second one is VotingEnsemble with accuracy was 0.91540212.

<img align="center" width="700" height="300" src="https://github.com/dokiem/Optimizing-an-ML-Pipeline-in-Azure/blob/main/images/AutoML-Best.png">

## Pipeline comparison

The best performing model was the AutoML model, we had best model ID "AutoML_6f7e7e95-7c34-432b-bd0b-2f521143bb03_0" with the accuracy was 0.91555387 and the algorithm used was MaxAbsScaler LightGBM.

<img align="center" width="700" height="300" src="https://github.com/dokiem/Optimizing-an-ML-Pipeline-in-Azure/blob/main/images/AutoML.png">

For the HyperDrive model with ID "HD_0cad047f-e345-4a80-b3c6-2a07f277c86e_3". It use a Scikit-learn Logistic Regression model, using HyperDrive to optimize hyperparameters. The accuracy is "0.9094082" with hyperparameters : {"--C": 0.7, "--max_iter": 60}

<img align="center" width="700" height="300" src="https://github.com/dokiem/Optimizing-an-ML-Pipeline-in-Azure/blob/main/images/HyperDrive.png">

Logistic Regression is simple, interpretable, and quick, but might need additional techniques to handle imbalanced data effectively.
LightGBM with MaxAbsScaler is powerful and handles imbalanced data more natively, making it a strong choice for large, complex datasets where performance is critical.

## Future work

We can see in the dataset that 11.2% of the labels are 'yes' and 88.8% are 'no'. This led to predicting better with label 0.
So, I think I can use resampling techniques or find some other method to deal with imbalanced data before applying AutoML.

## Proof of cluster clean up

<img align="center" width="700" height="300" src="https://github.com/dokiem/Optimizing-an-ML-Pipeline-in-Azure/blob/main/images/Cluster-Deleting.png">
