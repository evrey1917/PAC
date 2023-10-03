import pandas as pd
import numpy as np
import datetime as dt

frame1 = pd.read_csv('cinema_sessions.csv', index_col=0)
frame2 = pd.read_csv('titanic_with_labels.csv', index_col=0)

frame = pd.merge(frame1, frame2, on = 'check_number')
frame["session_start"] = frame["session_start"].astype("datetime64[ns]")

frame['sex'] = frame['sex'].map({'m':1,'M':1,'м':1,'М':1,'ж':0,'Ж':0})
frame = frame[frame['sex'].notna()]

frame['row_number'] = frame['row_number'].fillna(frame['row_number'].max(skipna = True))

frame['liters_drunk'] = frame['liters_drunk'].map(lambda x: None if x < 0 or x >= 7 else x)
frame['liters_drunk'] = frame['liters_drunk'].fillna(frame['liters_drunk'].mean(skipna = True))

frame[['age_child', 'age_adult', 'age_old']] = frame['age'].map(lambda x: str(x) + ' _ - _ -' if (x < 18) else '- _ ' + str(x) + ' _ -' if (x >= 18 and x <= 50) else '- _ - _ ' + str(x)).str.split('_', expand = True)
frame = frame.drop('age', axis = 1)

frame['drink'] = frame['drink'].map(lambda x: 1 if 'beer' in x else 0)

frame[['morning', 'day', 'evening']] = frame['session_start'].map(lambda x: str(x.time()) + ' _ - _ -' if (6 <= (x - dt.datetime.combine(dt.date.today(), dt.time.min)).seconds // 3600 < 12) else '- _ ' + str(x.time()) + ' _ -' if (12 <= (x - dt.datetime.combine(dt.date.today(), dt.time.min)).seconds // 3600 < 18) else '- _ - _ ' + str(x.time())).str.split('_', expand = True)
frame = frame.drop('session_start', axis = 1)
print(frame)
