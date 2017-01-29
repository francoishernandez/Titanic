# titanic_comparison.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

basis = pd.read_csv('survived.csv', sep=";")
data = basis.iloc[892:]
tocompare = pd.read_csv('my_solution_5.csv')
tocompare = tocompare.set_index('PassengerId')

tocompare['Test'] = data['survived']
tocompare['ID'] = tocompare.index

print(tocompare.groupby(['Survived', 'Test']).count())
