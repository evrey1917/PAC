import pandas as pd
import numpy as np

frame = pd.read_csv('wells_info.csv', index_col=0)

frame_mod2 = frame.set_index("BasinName")

delta = pd.Timedelta(days = 0, seconds = 0)
for i in frame["BasinName"].unique():
    print((abs(frame_mod2.loc[i]["CompletionDate"].astype("datetime64[ns]") - frame_mod2.loc[i]["FirstProductionDate"].astype("datetime64[ns]"))).sum().days // 30)
    
    # frame0 = frame_mod2.loc[i]
    # sum = (frame0["CompletionDate"].astype("datetime64[ns]").sum() - frame0["FirstProductionDate"].astype("datetime64[ns]").sum()).days
    # for j in range(len(frame0)):
    #     sum += abs((frame0["CompletionDate"].astype("datetime64[ns]")[j] - frame0["FirstProductionDate"].astype("datetime64[ns]")[j]).days)
    # print("Number of months for ", i, " in work: ", sum // 30)