import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

frame = pd.read_csv('wells_info_with_prod.csv', index_col=0)

frome = frame.reset_index()[["FirstProductionDate", "WATER_PER_FOOT", "Prod1Year"]]
frome["FirstProductionDate"] = pd.to_datetime(frome["FirstProductionDate"])
frome["FirstProductionDate"] = pd.to_numeric(frome["FirstProductionDate"])

scaler = MinMaxScaler()

X = frome                       #   features
y = frome["Prod1Year"]          #   label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#   fit_transform сразу фитит и трансформит, а чисто фит только формулу составляет, которая потом применяется в трансформе
scaler.fit(X_train, y_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print()
print(X_test)