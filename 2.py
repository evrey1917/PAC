import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score

def prepare_num(df):
    df_num = df.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    df_sex = pd.get_dummies(df['Sex'])
    df_emb = pd.get_dummies(df['Embarked'], prefix='Emb')
    df_pcl = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df_num = pd.concat((df_num, df_sex, df_emb, df_pcl), axis=1)
    return df_num

frame = pd.read_csv('train.csv', index_col=0)

X = frame.drop(["Survived", "Name", "Ticket", "Cabin"], axis=1)
y = frame["Survived"]

X = prepare_num(X)
X = X.fillna(X.median())
# print(X)

X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
X_test0, X_valid0, y_test, y_valid = train_test_split(X_test0, y_test, test_size=0.5, random_state=1)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train0)
X_test = scaler.transform(X_test0)
X_valid = scaler.transform(X_valid0)

modelRF = RandomForestClassifier(n_estimators=30, max_depth=4)
modelRF.fit(X_train0, y_train)

importances = modelRF.feature_importances_
features = X_train0.columns
# # Добавление сортировки по важности
indices = np.argsort(importances)
# print(features[indices])
two = features[indices][::-1][:2]
four = features[indices][::-1][:4]
eight = features[indices][::-1][:8]


# for 2 important features
X_train2 = scaler.fit_transform(X_train0[two.to_list()])
X_test2 = scaler.transform(X_test0[two.to_list()])
X_valid2 = scaler.transform(X_valid0[two.to_list()])

modelRF2 = RandomForestClassifier(n_estimators=30, max_depth=4)
modelRF2.fit(X_train2, y_train)

# for 4 important features
X_train4 = scaler.fit_transform(X_train0[four.to_list()])
X_test4 = scaler.transform(X_test0[four.to_list()])
X_valid4 = scaler.transform(X_valid0[four.to_list()])

modelRF4 = RandomForestClassifier(n_estimators=30, max_depth=4)
modelRF4.fit(X_train4, y_train)

# for 8 important features
X_train8 = scaler.fit_transform(X_train0[eight.to_list()])
X_test8 = scaler.transform(X_test0[eight.to_list()])
X_valid8 = scaler.transform(X_valid0[eight.to_list()])

modelRF8 = RandomForestClassifier(n_estimators=30, max_depth=4)
modelRF8.fit(X_train8, y_train)

# for all features
y_predict = modelRF.predict(X_valid0)
print("Random forest ALL features valid:", recall_score(y_valid, y_predict, average='weighted'))

y_predict = modelRF.predict(X_test0)
print("Random forest ALL features test:", recall_score(y_test, y_predict, average='weighted'))

# for 2 important features
y_predict = modelRF2.predict(X_valid2)
print("Random forest 2 features valid:", recall_score(y_valid, y_predict, average='weighted'))

y_predict = modelRF2.predict(X_test2)
print("Random forest 2 features test:", recall_score(y_test, y_predict, average='weighted'))

# for 4 important features
y_predict = modelRF4.predict(X_valid4)
print("Random forest 4 features valid:", recall_score(y_valid, y_predict, average='weighted'))

y_predict = modelRF4.predict(X_test4)
print("Random forest 4 features test:", recall_score(y_test, y_predict, average='weighted'))

# for 8 important features
y_predict = modelRF8.predict(X_valid8)
print("Random forest 8 features valid:", recall_score(y_valid, y_predict, average='weighted'))

y_predict = modelRF8.predict(X_test8)
print("Random forest 8 features test:", recall_score(y_test, y_predict, average='weighted'))