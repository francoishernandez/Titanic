# titanic_preprocessing.py

import pandas as pd
import numpy as np


# Import data from csv
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Create copies to preserve original datasets
train_copy = train.copy()
test_copy = test.copy()

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

# Fill missing values 'Fare'

train_copy['Fare'] = train_copy['Fare'].fillna(train_copy['Fare'].median())
test_copy['Fare'] = test_copy['Fare'].fillna(test_copy['Fare'].median())

# Fill missing values 'Embarked'
train_copy["Embarked"] = train_copy['Embarked'].fillna('S')
test_copy["Embarked"] = test_copy['Embarked'].fillna('S')

# Family Size

train_copy['FamilySize'] = train_copy['SibSp'] + train_copy['Parch'] + 1
test_copy['FamilySize'] = test_copy['SibSp'] + test_copy['Parch'] + 1

# Is Alone

train_copy['IsAlone'] = 0
train_copy.loc[train_copy['FamilySize']==1,'IsAlone'] = 1
test_copy['IsAlone'] = 0
test_copy.loc[test_copy['FamilySize']==1,'IsAlone'] = 1



# Export dataset in .csv files for easier inspection and further use
PassengerId =np.array(test["PassengerId"]).astype(int)
train_copy.to_csv("train_data.csv", index=False)
test_copy.to_csv("test_data.csv",index=False)
