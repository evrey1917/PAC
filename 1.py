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

X_megatest, X_test, y_megatest, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

scaler = MinMaxScaler()

y_train = y_megatest
X_train = scaler.fit_transform(X_megatest)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

modelRF = RandomForestClassifier(n_estimators=30, max_depth=4)
modelRF.fit(X_train, y_train)

# importances = modelRF.feature_importances_
# features = X_megatest.columns
# # # Добавление сортировки по важности
# indices = np.argsort(importances)

# plt.title('Важность признаков')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), features[indices])
# plt.xlabel('Относительная важность')
# plt.show()

y_predict = modelRF.predict(X_valid)
print("Random forest valid:", recall_score(y_valid, y_predict, average='weighted'))

y_predict = modelRF.predict(X_test)
print("Random forest test:", recall_score(y_test, y_predict, average='weighted'))

# np.display(modelRF.score(X_train, y_train))

# predict = model.predict(X_valid)

from xgboost import XGBClassifier

modelXGBC = XGBClassifier(n_estimators=20, max_depth=4)
modelXGBC.fit(X_train, y_train)
predict = modelXGBC.predict(X_test)

y_predict = modelXGBC.predict(X_valid)
print("Boost valid:", recall_score(y_valid, y_predict, average='weighted'))

y_predict = modelXGBC.predict(X_test)
print("Boost test:", recall_score(y_test, y_predict, average='weighted'))


from sklearn.linear_model import LogisticRegression
# modelLG = LogisticRegression(C=0.3, solver='lbfgs')
modelLG = LogisticRegression(C=0.1, solver='saga')
modelLG.fit(X_train, y_train)

y_predict = modelLG.predict(X_valid)
print("Linear valid:", recall_score(y_valid, y_predict, average='weighted'))

y_predict = modelLG.predict(X_test)
print("Linear test:", recall_score(y_test, y_predict, average='weighted'))


from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier()
modelKNN.fit(X_train, y_train)

y_predict = modelKNN.predict(X_valid)
print("KNN valid:", recall_score(y_valid, y_predict, average='weighted'))

y_predict = modelKNN.predict(X_test)
print("KNN test:", recall_score(y_test, y_predict, average='weighted'))