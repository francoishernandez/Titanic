# titanic_model2.py

# titanic_model.py

# Import libraries and tools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime



# Import sklearn tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn import cross_validation

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit

# Import data from csv

train = pd.read_csv('train_feat.csv')
test = pd.read_csv('test_feat.csv')

train_copy = train.copy()
test_copy = test.copy()


############################### MODEL CREATION ###############################

##################### FEATURE SELECTION #####################

features = [	"Pclass",
				"Sex",
			 	#"Age", 
			 	#"SibSp", 
			 	#"Parch",
			 	#"Fare", 
			 	"Embarked",
			 	"Age Group",
			 	#"Child",
			 	#"Baby",
			 	"Title",
			 	#"Fare Group",
			 	#"FamilySize",
			 	#"IsAlone"
			 	]



# Create the target
target = train_copy['Survived'].values
features_model = train_copy[features].values
train_copy[features] = train_copy[features].astype(float)

X = train_copy[features].values
y = train_copy['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size = 0.2, random_state = 42)

model = SVC(kernel = 'linear', C=1)
classifier = model.fit(X_train, y_train)

shuffle_validator = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
def test_classifier(clf):
	scores = cross_validation.cross_val_score(clf, X, y, cv=shuffle_validator)
	print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))

#test_classifier(classifier)

predict = classifier.predict(X_test)

# LEARNING CURVE

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
	n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
	"""
	Generate a simple plot of the test and training learning curve.

	Parameters
	----------
	estimator : object type that implements the "fit" and "predict" methods
	An object of that type which is cloned for each validation.

	title : string
	Title for the chart.

	X : array-like, shape (n_samples, n_features)
	Training vector, where n_samples is the number of samples and
	n_features is the number of features.

	y : array-like, shape (n_samples) or (n_samples, n_features), optional
	Target relative to X for classification or regression;
	None for unsupervised learning.

	ylim : tuple, shape (ymin, ymax), optional
	Defines minimum and maximum yvalues plotted.

	cv : int, cross-validation generator or an iterable, optional
	Determines the cross-validation splitting strategy.
	Possible inputs for cv are:
	  - None, to use the default 3-fold cross-validation,
	  - integer, to specify the number of folds.
	  - An object to be used as a cross-validation generator.
	  - An iterable yielding train/test splits.

	For integer/None inputs, if ``y`` is binary or multiclass,
	:class:`StratifiedKFold` used. If the estimator is not a classifier
	or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

	Refer :ref:`User Guide <cross_validation>` for the various
	cross-validators that can be used here.

	n_jobs : integer, optional
	Number of jobs to run in parallel (default 1).
	"""
	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(
	estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
	 train_scores_mean + train_scores_std, alpha=0.1,
	 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
	 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
	 label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
	 label="Cross-validation score")

	plt.legend(loc="best")
	return plt

title = "Learning Curves (RandomForest)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = RandomForestClassifier()
#plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=1)
plot_learning_curve(estimator, title, X, y, (0.3, 1.01), cv=cv, n_jobs=4)

title = "Learning Curves (Decision Tree)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = DecisionTreeClassifier()
#plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

def plot_validation_curve(estimator, title, X, y, ylim=None, cv=None,
	n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
	"""
	Generate a simple plot of the test and training learning curve.

	Parameters
	----------
	estimator : object type that implements the "fit" and "predict" methods
	An object of that type which is cloned for each validation.

	title : string
	Title for the chart.

	X : array-like, shape (n_samples, n_features)
	Training vector, where n_samples is the number of samples and
	n_features is the number of features.

	y : array-like, shape (n_samples) or (n_samples, n_features), optional
	Target relative to X for classification or regression;
	None for unsupervised learning.

	ylim : tuple, shape (ymin, ymax), optional
	Defines minimum and maximum yvalues plotted.

	cv : int, cross-validation generator or an iterable, optional
	Determines the cross-validation splitting strategy.
	Possible inputs for cv are:
	  - None, to use the default 3-fold cross-validation,
	  - integer, to specify the number of folds.
	  - An object to be used as a cross-validation generator.
	  - An iterable yielding train/test splits.

	For integer/None inputs, if ``y`` is binary or multiclass,
	:class:`StratifiedKFold` used. If the estimator is not a classifier
	or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

	Refer :ref:`User Guide <cross_validation>` for the various
	cross-validators that can be used here.

	n_jobs : integer, optional
	Number of jobs to run in parallel (default 1).
	"""
	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("$\gamma$")
	plt.ylabel("Score")

	param_range = np.logspace(-6,5)

	#train_scores, test_scores = validation_curve(estimator, X, y, param_name="gamma", param_range=param_range, cv=cv, scoring="accuracy", n_jobs=n_jobs)
	train_scores, test_scores = validation_curve(estimator, X, y, param_name = "gamma", param_range=param_range, cv=cv, scoring="accuracy", n_jobs=n_jobs)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()
	lw = 2
	plt.semilogx(param_range, train_scores_mean, label="Training score",
		 color="darkorange", lw=lw)
	plt.fill_between(param_range, train_scores_mean - train_scores_std,
		 train_scores_mean + train_scores_std, alpha=0.2,
		 color="darkorange", lw=lw)
	plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
		 color="navy", lw=lw)
	plt.fill_between(param_range, test_scores_mean - test_scores_std,
		 test_scores_mean + test_scores_std, alpha=0.2,
		 color="navy", lw=lw)
	plt.legend(loc="best")

	return plt

title = "Validation Curves (SVM, RBF kernel, $\gamma$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC()
plot_validation_curve(estimator, title, X, y, (0.3, 1.01), cv=cv, n_jobs=4)


plt.show()


