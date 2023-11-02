import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import recall_score

frame = pd.read_csv('titanic_prepared.csv', index_col=0)

X = frame.drop("label", axis=1)
y = frame["label"]

scaler = MinMaxScaler()

X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

X_train = scaler.fit_transform(X_train0)
X_test = scaler.transform(X_test0)

modelTree = DecisionTreeClassifier(max_depth=7, criterion='entropy')
modelTree.fit(X_train, y_train)

y_predict = modelTree.predict(X_test)
print("Tree accuracy:", recall_score(y_test, y_predict, average='weighted'))


from sklearn.linear_model import LogisticRegression
modelLG = LogisticRegression(C=0.1, solver='lbfgs')
modelLG.fit(X_train, y_train)

y_predict = modelLG.predict(X_test)
print("Log. regression accuracy:", recall_score(y_test, y_predict, average='weighted'))


from xgboost import XGBClassifier
modelXGBC = XGBClassifier(n_estimators=20, max_depth=4)
modelXGBC.fit(X_train, y_train)

y_predict = modelXGBC.predict(X_test)
print("Boost accuracy:", recall_score(y_test, y_predict, average='weighted'))


importances = modelTree.feature_importances_
features = X_train0.columns
indices = np.argsort(importances)

X_train2 = X_train0[features[indices][::-1][:2]]
X_test2 = X_test0[features[indices][::-1][:2]]

modelTree2 = DecisionTreeClassifier(max_depth=3, criterion='entropy')
modelTree2.fit(X_train2, y_train)

y_predict = modelTree2.predict(X_test2)
print("Tree 2 features accuracy:", recall_score(y_test, y_predict, average='weighted'))