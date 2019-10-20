# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

sns.set()

df = pd.read_csv("train.csv")  #total is 8523
# print(df.head())
# df.info()

#sns.catplot('Item_Type','Item_Outlet_Sales',data=df)
#sns.catplot('Item_Weight','Item_Outlet_Sales',data=df)
#sns.catplot('Item_MRP','Item_Outlet_Sales',data=df)
#sns.catplot('Item_Visibility','Item_Outlet_Sales ',data=df)
#sns.catplot('Item_Type','Item_Outlet_Sales',data=df)
#sns.catplot('Item_Fat_Content','Item_Outlet_Sales',data=df)

# Fixing the Item Weights
df2 = df['Item_Weight'].isnull()
gf = df.groupby('Item_Identifier')['Item_Weight'].mean()

for i in df[df2].index:
    df.loc[i,"Item_Weight"] = gf[df.iloc[i]["Item_Identifier"]]
   # print(gf[df.iloc[i]["Item_Identifier"]])
#print(df[df["Item_Weight"].isnull()])


# Fixing the Outlet Size
#print("Fixing Outlet Size")
df2 = df['Outlet_Size'].isnull()

gf = df.groupby('Outlet_Identifier')['Outlet_Size'].apply(lambda x: x.mode())

#print(gf)


for i in df[df2].index:
    if df.loc[i,"Outlet_Identifier"] in gf.index:
        df.loc[i,"Outlet_Size"] = gf[df.iloc[i]["Outlet_Identifier"]]
    else:
        df.loc[i,"Outlet_Size"] = "Unknown"
   # print(gf[df.iloc[i]["Item_Identifier"]])

#print(df[df["Outlet_Size"].isnull()])
#print(df.info())


df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'})

   
#print(df['Item_Fat_Content'].unique())

df['Item_Type_Code'] = df['Item_Identifier'].str[:2]
df['Item_Type_Code'] = df['Item_Type_Code'].replace({'FD':'Food','DR':'Drink','NC':'Non Consumable'})
#print(df['Item_Type_Code']) 



#Zero'ing fat content in non edibles
df.loc[df['Item_Type_Code']=='Non Consumable','Item_Fat_Content'] = "N/A"


# Replace NAN's in Item_Weight to Zero, the last remaining 4.
df['Item_Weight'].fillna(value=0, inplace=True)
#df.info()


#Convert Outlet_Establishment_Year into Years_of_Establishment
df['Years_Of_Establishment'] = 2019 - df['Outlet_Establishment_Year']

# Checking for zero items within int/float columns
for column in df.columns:
    if (df[column].dtypes == "float64") or (df[column].dtypes == "int64"):
         pass

# Fixing the Zero items in item visibility
gf = df.groupby('Item_Type')['Item_Visibility'].mean()
df2 = df[df['Item_Visibility']==0]
#print(df2)

for i in df2.index:
    df.loc[i,"Item_Visibility"] = gf[df.iloc[i]["Item_Type"]]
   # print(gf[df.iloc[i]["Item_Identifier"]])
#print(df[df["Item_Visibility"]==0])

df.drop(['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Establishment_Year'], axis=1,inplace=True)


# One Hot Encoding
df = pd.get_dummies(df,columns=['Outlet_Size','Outlet_Location_Type','Item_Type_Code','Item_Fat_Content','Outlet_Type'])
#print(df.columns)
df.info()

# Check on clean dataset, and make sure it is really clean
# df.to_csv('Cleaned_Output.csv')



# Train_Test_Splitting the data
# Check on train_test_split in scikitlearn, and use it to split into train and cross validation
X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:,:3].join(df.iloc[:,4:]), df.iloc[:,3:4], test_size=.33, random_state=99)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(df.shape)
X_all = df.iloc[:,:3].join(df.iloc[:,4:])
y_all = df.iloc[:,3:4]


# Manual Linear Regression variance/bias testing
reg = LinearRegression().fit(X_train, y_train)
Y_pred = reg.predict(X_test)
Y_error = Y_pred - y_test
Y_error = np.square(Y_error)
Y_error_mean = np.mean(Y_error)
Y_error_mean_sqrt = np.sqrt(Y_error_mean)

Y_pred_train = reg.predict(X_train)
Y_error_train = Y_pred_train - y_train
Y_error_train = np.square(Y_error_train)
Y_error_mean_train = np.mean(Y_error_train)
Y_error_mean_sqrt_train = np.sqrt(Y_error_mean_train)


print(Y_error_mean_train)
print(Y_error_mean)
print(Y_error_mean_sqrt_train)
print(Y_error_mean_sqrt)





# Learning Curve Analysis
from sklearn.learning_curve import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
     LinearRegression(), X_all, y_all, train_sizes=[50, 100, 300, 1000, 2000, 3000, 4000, 5682],cv=3, scoring = 'neg_mean_squared_error')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

print(train_scores_mean, test_scores_mean)

# Plotting the Learning Curve
plt.figure(figsize=(20,10))
plt.grid()
plt.title("Leaning Curve, Linear Regression")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.plot(train_sizes, train_scores_mean)
plt.plot(train_sizes, test_scores_mean)


"""
To Dos:
    Check on clean dataset, and make sure it is really clean
    Check on train_test_split in scikitlearn, and use it to split into train and cross validation
    Draw the bias / variance curve, first run it for 1000 data, 3000 data, 8000 data.
    Run Linear Regression on the train set, then the cross validation set
    Follow the same process as our regressions in the previous examples. 
"""



"""
# To Do's
# Fix the Fat part, #Reduce the number of category in ITem Types, #Fix last of Item Weights
# Fat content in non edibles does not make sense, make it zero.
# Replace NAN's in Item_Weight to Zero, the last remaining 4. 
#After this , we will work on it Monday.
"""