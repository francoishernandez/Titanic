# Titanic Kaggle - RandomForest

# Titanic Kaggle competition

# Import libraries and tools

import pandas as pd
import numpy as np

from sklearn import tree

# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# Import data from csv

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Display data

print(train.head())
print(test.head())

# Create copies to preserve original datasets
train_copy = train.copy()
test_copy = test.copy()

print(train['Survived'][train['Sex'] == 'male'].value_counts())

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


# Create the target and features numpy arrays: target, features_one
target = train_copy['Survived'].values
features_one = train_copy[["Pclass", "Sex", "Age", "Fare"]].values

#print(train_copy['Survived'])

# Write your solution to a csv file with the name my_solution.csv
PassengerId =np.array(test["PassengerId"]).astype(int)
features = ["Pclass", "Sex", "Age", "Fare"]
train_copy[features].to_csv("train_data.csv", index_label = ["PassengerId"])



# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
