# titanic_model.py

# Import libraries and tools

import pandas as pd
import numpy as np

import datetime



# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Import data from csv

train = pd.read_csv('train_feat.csv')
test = pd.read_csv('test_feat.csv')

train_copy = train.copy()
test_copy = test.copy()


############################### MODEL CREATION ###############################

# Create the target
target = train_copy['Survived'].values

##################### FEATURE SELECTION #####################

features = [	"Pclass",
				"Sex",
			 	#"Age", 
			 	#"SibSp", 
			 	#"Parch",
			 	#"Fare", 
			 	"Embarked",
			 	#"Age Group",
			 	#"Child",
			 	#"Baby",
			 	"Title",
			 	"Fare Group",
			 	"FamilySize",
			 	"IsAlone"
			 	]


#select_feat = SelectKBest(chi2,k=5).fit_transform(train_copy[features].values,target)
#print("SELECTED FEATURES WITH SKLEARN")
#print(select_feat.shape)

features_model = train_copy[features].values
train_copy[features] = train_copy[features].astype(float)

print(train_copy[features].head())

# Building and fitting my_forest

# Random Forest
#model = RandomForestClassifier(max_depth = 12, min_samples_split=2, n_estimators = 100, random_state = 1)
#model = RandomForestClassifier()

# Decision Tree
#model = DecisionTreeClassifier(max_depth=12, min_samples_split=2, min_samples_leaf=2, presort=True)
#model = DecisionTreeClassifier(max_depth=10)

# K Neighbors
#model = KNeighborsClassifier()

# SVM
model = SVC(probability=True)


my_classif = model.fit(features_model, target)

# Print the score of the fitted random forest
print(my_classif.score(features_model, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test_copy[features].values
pred_classif = my_classif.predict(test_features)
print(len(pred_classif))

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test_copy["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_classif, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

#print(features)
#print(my_classif.feature_importances_)

# Write your solution to a csv file with the name my_solution.csv
date = str(datetime.datetime.now())
my_solution.to_csv("my_solution_"+date+str(features)+".csv", index_label = ["PassengerId"])

basis = pd.read_csv('survived.csv', sep=";")
data = basis.iloc[892:]
tocompare = pd.read_csv("my_solution_"+date+str(features)+".csv")
tocompare = tocompare.set_index('PassengerId')

tocompare['Test'] = data['survived']
tocompare['ID'] = tocompare.index
match = tocompare['Test'] == tocompare['Survived']
tocompare['Match'] = 0
tocompare.loc[match,'Match'] = 1
print(tocompare['Match'].mean())

print(tocompare.groupby(['Survived', 'Test']).count())
