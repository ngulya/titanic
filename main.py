import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import warnings
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.simplefilter('ignore')

def return_embarked(s):
	if s['Embarked'] == -1:
		return int(train['Embarked'][train['Pclass'] == s['Pclass']].median())
	return s['Embarked']

def return_fare(s):
	if s['Fare'] == -1:
		return int(train['Fare'][train['Fare'] == s['Fare']].median() / 7)
	return int(s['Fare'] / 7)

def return_married(s):

	st = s['Name'].lower()
	if s['Sex'] == 'male':
		s['Sex'] = 0
	elif st.find('miss') == -1:
		s['Sex'] = 1
	else:
		s['Sex'] = 2 
	return s

def return_ticket(s):
	if s['Ticket'] == 'no':
		return 'no'
	s['Ticket'] = s['Ticket'].lower()
	st = s['Ticket'].upper().split(' ')
	return st[0][0]


##MAIN
if os.path.exists("doc/newtrain.csv") and os.path.exists("doc/newtest.csv"):
	train = pd.read_csv('doc/newtrain.csv')
	test = pd.read_csv('doc/newtest.csv')
else:
	train = pd.read_csv('doc/train.csv')
	test = pd.read_csv('doc/test.csv')
	train = train.append(test)

	train = train.reindex(columns=('PassengerId','Survived', 'Pclass', 'Name','Sex','Age','SibSp', 
		'Parch', 'Ticket', 'Fare','Cabin','Embarked'), fill_value=0)

	indxm = train['Sex'] == 'male'
	indxf = train['Sex'] == 'female'
	train['Age'][indxm] = train['Age'][indxm].fillna(train['Age'][indxm].median())
	train['Age'][indxf] = train['Age'][indxf].fillna(train['Age'][indxf].median() )
	train['Age'] = (train['Age'] / 10).astype(int) * 10

	train['Embarked'][train['Embarked'] == 'C'] = 1
	train['Embarked'][train['Embarked'] == 'Q'] = 2
	train['Embarked'][train['Embarked'] == 'S'] = 3
	train['Embarked'] = train['Embarked'].fillna(-1)
	train['Embarked'] = train.apply(return_embarked, axis = 1)
	
	train['Fare'] = train['Fare'].fillna(-1)
	train['Fare'] = train.apply(return_fare, axis = 1)
	
	train = train.apply(return_married, axis = 1)
	
	le = LabelEncoder()
	train['Ticket'] = train['Ticket'].fillna('no')
	train['Typeticket'] = train.apply(return_ticket, axis = 1)
	le.fit(train['Typeticket'])
	train['Typeticket'] = le.transform(train['Typeticket'])

	train = train.drop(['Ticket', 'Cabin', 'Name'], axis = 1)
	mask = train['PassengerId'] < 892
	train, test = train[mask], train[~mask]

	train.to_csv('doc/newtrain.csv', index = False)
	test.to_csv('doc/newtest.csv', index = False)

	Grph = train.groupby(['Fare', 'Survived']).size().unstack().fillna(0)
	Grph.plot(kind='bar');
	plt.show()

	Grph = train.groupby(['Sex', 'Survived']).size().unstack().fillna(0)
	Grph.plot(kind='bar', title='0 - Mister, 1 - Misses, 2 - Miss');
	plt.show()

	Grph = train.groupby(['Age', 'Survived']).size().unstack().fillna(0)
	Grph.plot(kind='bar');
	plt.show()

	Grph = train.groupby(['Typeticket', 'Survived']).size().unstack().fillna(0)
	Grph.plot(kind='bar');
	plt.show()

submit = pd.read_csv('doc/gender_submission.csv')
target_train = train['Survived']
train = train.drop(['Survived', 'PassengerId'], axis = 1)
test = test.drop(['Survived', 'PassengerId'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(train, target_train,test_size =  0.25, random_state=0)
Y_test = Y_test.values.ravel()
Y_train = Y_train.values.ravel()

modelX = xgb.XGBClassifier(learning_rate = 0.01, max_depth = 3, n_estimators = 250)
modelX.fit(X_train, Y_train)
res = modelX.predict(X_test).astype(int)
print("accuracy score = %f"% accuracy_score(Y_test, res))
print("MSE = %f"% mean_squared_error(Y_test, res))

result = modelX.predict(test).astype(int)
submit['Survived'] = result
submit.to_csv('./submit.csv', index = False)