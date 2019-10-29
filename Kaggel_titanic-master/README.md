# Kaggel_titanic
In this project, an analysis of what sorts of people were likely to survive the titanic  tragedy is done by applying the tools of machine learning to predict which passengers survived the tragedy.

I have used Logistic regression and SVM for the classification.

### Dataset details
Download the data set [here](https://www.kaggle.com/c/titanic/data) 

The data has been split into two groups:

--training set (train.csv) - The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

--test set (test,csv) - The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

Attributes:
1. Name
2. Age 
3. PassengerId
4. Pclass
5. Sex
6. SibSp
7. Parch
8. Ticket
9. Fare
10. Cabin
11. Embarked

### Code Details
A description on files in this repository:
1. **data_vis.py** : Data visulaization. Mostly discuss how the number of survivor vary for different attributes.
2. **main.py**: The following classification algorithms are used 
-- Logistic Regression 
-- SVM
The algorithms are compared using precision, recall and accuracy.

##**REPORT**
### Dataset Visualisation
#### check the number of survivors and non survivors. 
![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/Figure_1.png) 

#### Dependency of survival on sex
![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/Figure_3.png) 

#### Dependency of survival on ticket class
![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/Figure_4.png)
![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/Figure_5.png) 

#### Dependency on port of embarkment
![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/Figure_7.png) 
![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/Figure_8.png)
![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/Figure_9.png)

#### Dependency on Age
![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/Figure_10.png) 
![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/Figure_11.png) 
 
#### Dependency on no of siblings onboard
![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/Figure_12.png)

#### Dependency on presence of parent/ children on board 
![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/Figure_13.png) 

#### Dependency on ticket fare
![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/Figure_14.png) 

### Algorithm comparision
The dataset was split into train and test dataset. 30% of dataset was randomly selected and used for cross validation. depending on the comparision between predicted category and original output, the following result was achieved. 

1. **Logistic regression**

![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/log_reg.png) 

2. **SVM**

![](https://github.com/sumanp31/Kaggel_titanic/blob/master/Plot/svc.png) 


### Algorithm output
The coefficient calculated using the above algorithms were than predict the survivour for Test set. The output for each algorithm can be found here:
1. **Logistic regression** : [log_reg.csv](https://github.com/sumanp31/Kaggel_titanic/blob/master/output/log_reg.csv) 
2. **SVM** : [svc.csv](https://github.com/sumanp31/Kaggel_titanic/blob/master/output/svc.csv) 