# Titanic Kaggle - Input handling

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

#print(train.head())
#print(test.head())

# Create copies to preserve original datasets
train_copy = train.copy()
test_copy = test.copy()

#print(train['Survived'][train['Sex'] == 'male'].value_counts())


############################### DATASET CLEANING ###############################

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


# Export dataset in .csv files for easier inspection
PassengerId =np.array(test["PassengerId"]).astype(int)
features = ["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]
train_copy[features].to_csv("train_data.csv", index_label = ["PassengerId"])
test_copy[features].to_csv("test_data.csv", index_label = ["PassengerId"])


############################### MODEL CREATION ###############################

# Create the target
target = train_copy['Survived'].values

##################### FEATURE SELECTION #####################

# Get the user to choose features

train_features = list(train_copy.columns)
train_features.remove('PassengerId')
train_features.remove('Survived')
print(train_features)

print("Which features to consider? \n")

feature_input = "blabla"
features = []
while(feature_input <> "END"):
	print("Choose a feature in the following list (press Enter to validate, write END to finish)")
	print(train_features)
	feature_input = raw_input()
	if(feature_input in train_features):
		features.append(feature_input)
		print("Selected features : ")
		print(features)
		train_features.remove(feature_input)
		print("Available features : ")
		print(train_features)
	else:
		print("Invalid feature")

print("Selected features are :")
print(features)

##################### MODEL SELECTION #####################






