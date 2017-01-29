# titanic_features.py

import pandas as pd
import numpy as np

# Import data from csv
train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

train_copy = train.copy()
test_copy = test.copy()


# Create column 'Age Group'

train_copy['Age Group'] = float('NaN')
test_copy['Age Group'] = float('NaN')


row_adult = train_copy['Age'] <= 60
train_copy.loc[row_adult, 'Age Group'] = 2

row_child = train_copy['Age'] < 18
train_copy.loc[row_child, 'Age Group'] = 1

row_senior = train_copy['Age'] > 60
train_copy.loc[row_senior, 'Age Group'] = 3

row_baby = train_copy['Age'] < 2
train_copy.loc[row_baby, 'Age Group'] = 0

row_adult = test_copy['Age'] <= 60
test_copy.loc[row_adult, 'Age Group'] = 2

row_child = test_copy['Age'] < 18
test_copy.loc[row_child, 'Age Group'] = 1

row_senior = test_copy['Age'] > 60
test_copy.loc[row_senior, 'Age Group'] = 3

row_baby = test_copy['Age'] < 2
test_copy.loc[row_baby, 'Age Group'] = 0


train_copy['Age Group'] = train_copy['Age Group'].astype(int)
test_copy['Age Group'] = test_copy['Age Group'].astype(int)

# Create column 'Child'

train_copy['Child'] = float('NaN')
test_copy['Child'] = float('NaN')

row_child = train_copy['Age'] < 17
row_adult = train_copy['Age'] >=17
train_copy.loc[row_child,'Child'] = 1
train_copy.loc[row_adult,'Child'] = 0

row_child = test_copy['Age'] < 17
row_adult = test_copy['Age'] >=17
test_copy.loc[row_child,'Child'] = 1
test_copy.loc[row_adult,'Child'] = 0

train_copy['Child'] = train_copy['Child'].astype(int)
test_copy['Child'] = test_copy['Child'].astype(int)

# Create column 'Baby'

train_copy['Baby'] = float('NaN')
test_copy['Baby'] = float('NaN')

row_baby = train_copy['Age'] < 2
row_nb = train_copy['Age'] >= 2
train_copy.loc[row_baby,'Baby'] = 1
train_copy.loc[row_nb,'Baby'] = 0

row_baby = test_copy['Age'] < 2
row_nb = test_copy['Age'] >= 2
test_copy.loc[row_baby,'Baby'] = 1
test_copy.loc[row_nb,'Baby'] = 0

train_copy['Baby'] = train_copy['Baby'].astype(int)
test_copy['Baby'] = test_copy['Baby'].astype(int)


# Convert the 'Embarked' classes to integer form
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

# Create colum 'Title'

train_copy['Title'] = train_copy['Name'].str.split(', ').str[1].str.split(' ').str[0]
row_mr = train_copy['Title'] == 'Mr.'
row_mrs = train_copy['Title'] == 'Mrs.'
row_mt = train_copy['Title'] == 'Master.'
row_ms = train_copy['Title'] == 'Miss.'
row_rev = train_copy['Title'] == 'Rev.'

train_copy['Title'] = 0
train_copy.loc[row_mr, 'Title'] = 1
train_copy.loc[row_mrs, 'Title'] = 2
train_copy.loc[row_mt, 'Title'] = 3
train_copy.loc[row_ms, 'Title'] = 4
train_copy.loc[row_rev, 'Title'] = 5

test_copy['Title'] = test_copy['Name'].str.split(', ').str[1].str.split(' ').str[0]
row_mr = test_copy['Title'] == 'Mr.'
row_mrs = test_copy['Title'] == 'Mrs.'
row_mt = test_copy['Title'] == 'Master.'
row_ms = test_copy['Title'] == 'Miss.'
row_rev = test_copy['Title'] == 'Rev.'

test_copy['Title'] = 0
test_copy.loc[row_mr, 'Title'] = 1
test_copy.loc[row_mrs, 'Title'] = 2
test_copy.loc[row_mt, 'Title'] = 3
test_copy.loc[row_ms, 'Title'] = 4
test_copy.loc[row_rev, 'Title'] = 5



# Create column 'Fare Group'

train_copy['Fare Group'] = float('NaN')
test_copy['Fare Group'] = float('NaN')

row_low = train_copy['Fare'] < 35
train_copy.loc[row_low, 'Fare Group'] = 1

row_cheap = train_copy['Fare'] < 11
train_copy.loc[row_cheap, 'Fare Group'] = 0

row_exp = train_copy['Fare'] >= 35
train_copy.loc[row_exp, 'Fare Group'] = 2

row_low = test_copy['Fare'] < 35
test_copy.loc[row_low, 'Fare Group'] = 1

row_cheap = test_copy['Fare'] < 11
test_copy.loc[row_cheap, 'Fare Group'] = 0

row_exp = test_copy['Fare'] >= 35
test_copy.loc[row_exp, 'Fare Group'] = 2

train_copy['Fare Group'] = train_copy['Fare Group'].astype(int)
test_copy['Fare Group'] = test_copy['Fare Group'].astype(int)

# Family Size

train_copy['FamilySize'] = train_copy['SibSp'] + train_copy['Parch'] + 1
test_copy['FamilySize'] = test_copy['SibSp'] + test_copy['Parch'] + 1

# Is Alone

train_copy['IsAlone'] = 0
train_copy.loc[train_copy['FamilySize']==1,'IsAlone'] = 1
test_copy['IsAlone'] = 0
test_copy.loc[test_copy['FamilySize']==1,'IsAlone'] = 1

# Normalize

train_copy['Embarked'] = train_copy['Embarked']/max(train_copy['Embarked'])
test_copy['Embarked'] = test_copy['Embarked']/max(test_copy['Embarked'])

train_copy['Title'] = train_copy['Title']/max(train_copy['Title'])
test_copy['Title'] = test_copy['Title']/max(test_copy['Title'])

train_copy['FamilySize'] = train_copy['FamilySize']/max(train_copy['FamilySize'])
test_copy['FamilySize'] = test_copy['FamilySize']/max(test_copy['FamilySize'])

train_copy['Fare Group'] = train_copy['Fare Group']/max(train_copy['Fare Group'])
test_copy['Fare Group'] = test_copy['Fare Group']/max(test_copy['Fare Group'])

train_copy['Age Group'] = train_copy['Age Group']/max(train_copy['Age Group'])
test_copy['Age Group'] = test_copy['Age Group']/max(test_copy['Age Group'])



# Export dataset in .csv files for easier inspection and further use
PassengerId =np.array(test["PassengerId"]).astype(int)
train_copy.to_csv("train_feat.csv", index=False)
test_copy.to_csv("test_feat.csv", index=False)


