import pandas as pd
import numpy as np

frame = pd.read_csv('wells_info_na.csv', index_col=0)
frame["CompletionDate"] = frame["CompletionDate"].astype("datetime64[ns]")
frame["FirstProductionDate"] = frame["FirstProductionDate"].astype("datetime64[ns]")

numeric_list = frame.select_dtypes(include = ['number', 'datetime64']).keys()
frame[numeric_list] = frame[numeric_list].fillna(frame[numeric_list].mean())

other_list = frame.select_dtypes(exclude = ['number', 'datetime64']).keys()

frame[other_list] = frame[other_list].fillna(frame[other_list].mode().iloc[0])
print(frame)

# frame["LatWGS84"] = frame["LatWGS84"].fillna(frame["LatWGS84"].mean())
# frame["LonWGS84"] = frame["LonWGS84"].fillna(frame["LonWGS84"].mean())
# frame["PROP_PER_FOOT"] = frame["PROP_PER_FOOT"].fillna(frame["PROP_PER_FOOT"].mean())
# frame["CompletionDate"] = frame["CompletionDate"].fillna(frame["CompletionDate"].mean())
# frame["FirstProductionDate"] = frame["FirstProductionDate"].fillna(frame["FirstProductionDate"].mean())

# frame["formation"] = frame["formation"].fillna(frame["formation"].value_counts()[0])
# frame["BasinName"] = frame["BasinName"].fillna(frame["BasinName"].value_counts()[0])
# frame["StateName"] = frame["StateName"].fillna(frame["StateName"].value_counts()[0])
# frame["CountyName"] = frame["CountyName"].fillna(frame["CountyName"].value_counts()[0])

# print(frame)