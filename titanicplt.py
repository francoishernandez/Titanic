# Titanic Kaggle - RandomForest - Play with features

# Titanic Kaggle competition

# Import libraries and tools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import datasets, svm

# Import data from csv

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Display data

#print(train.head())
#print(test.head())

# Create copies to preserve original datasets
train_copy = train.copy()
test_copy = test.copy()

#print(train['Survived'][train['Sex'] == 'male'].value_counts())

# Convert the male and female groups to integer form

row_male = train_copy['Sex'] == 'male'
row_female = train_copy['Sex'] == 'female'

train_copy.loc[row_male,'Sex'] = 0
train_copy.loc[row_female,'Sex'] = 1

row_male = test_copy['Sex'] == 'male'
row_female = test_copy['Sex'] == 'female'

test_copy.loc[row_male,'Sex'] = 0
test_copy.loc[row_female,'Sex'] = 1

# Fill missing values 'Age'

train_copy['Age'] = train_copy['Age'].fillna(train_copy['Age'].median())
test_copy['Age'] = test_copy['Age'].fillna(test_copy['Age'].median())

# Create column Child and assign to 'NaN'

train_copy['Child'] = float('NaN')
test_copy['Child'] = float('NaN')

row_child = train_copy['Age'] < 18
row_adult = train_copy['Age'] >= 18

train_copy.loc[row_child, 'Child'] = 1
train_copy.loc[row_adult, 'Child'] = 0

row_child = test_copy['Age'] < 18
row_adult = test_copy['Age'] >= 18

test_copy.loc[row_child, 'Child'] = 1
test_copy.loc[row_adult, 'Child'] = 0

# Impute the Embarked variable
train_copy["Embarked"] = train_copy['Embarked'].fillna('S')
test_copy["Embarked"] = test_copy['Embarked'].fillna('S')

# Convert the Embarked classes to integer form
row_s = train_copy['Embarked'] == 'S'
row_c = train_copy['Embarked'] == 'C'
row_q = train_copy['Embarked'] == 'Q'

train_copy.loc[row_s, 'Embarked'] = 0
train_copy.loc[row_c, 'Embarked'] = 1
train_copy.loc[row_q, 'Embarked'] = 2

row_s = test_copy['Embarked'] == 'S'
row_c = test_copy['Embarked'] == 'C'
row_q = test_copy['Embarked'] == 'Q'

test_copy.loc[row_s, 'Embarked'] = 0
test_copy.loc[row_c, 'Embarked'] = 1
test_copy.loc[row_q, 'Embarked'] = 2

train_copy['Fare'] = train_copy['Fare'].fillna(train_copy['Fare'].median())
test_copy['Fare'] = test_copy['Fare'].fillna(test_copy['Fare'].median())

# Create a new feature Family Size
train_copy["family_size"] = train_copy['SibSp'] + train_copy['Parch'] + 1
test_copy["family_size"] = test_copy['SibSp'] + test_copy['Parch'] + 1

# Select features to consider
features = ["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "family_size" "Embarked"]

# Simple plot test



# specifies the parameters of our graphs
fig = plt.figure(figsize=(18,6), dpi=1600) 
alpha=alpha_scatterplot = 0.2 
alpha_bar_chart = 0.55

# lets us plot many diffrent shaped graphs together 
ax1 = plt.subplot2grid((2,3),(0,0))
# plots a bar graph of those who surived vs those who did not.               
train_copy.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
# this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1
ax1.set_xlim(-1, 2)
# puts a title on our graph
plt.title("Distribution of Survival, (1 = Survived)")    

plt.subplot2grid((2,3),(0,1))
plt.scatter(train_copy.Survived, train_copy.Age, alpha=alpha_scatterplot)
# sets the y axis lable
plt.ylabel("Age")
# formats the grid line style of our graphs                          
plt.grid(b=True, which='major', axis='y')  
plt.title("Survival by Age,  (1 = Survived)")

ax3 = plt.subplot2grid((2,3),(0,2))
train_copy.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
ax3.set_ylim(-1, len(train_copy.Pclass.value_counts()))
plt.title("Class Distribution")

plt.subplot2grid((2,3),(1,0), colspan=2)
# plots a kernel density estimate of the subset of the 1st class passangers's age
train_copy.Age[train_copy.Pclass == 1].plot(kind='kde')    
train_copy.Age[train_copy.Pclass == 2].plot(kind='kde')
train_copy.Age[train_copy.Pclass == 3].plot(kind='kde')
 # plots an axis lable
plt.xlabel("Age")    
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

ax5 = plt.subplot2grid((2,3),(1,2))
train_copy.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax5.set_xlim(-1, len(train_copy.Embarked.value_counts()))
# specifies the parameters of our graphs
plt.title("Passengers per boarding location")

