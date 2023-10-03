import pandas as pd
import numpy as np

frame1 = pd.read_csv('cinema_sessions.csv', index_col=0)
frame2 = pd.read_csv('titanic_with_labels.csv', index_col=0)

frame = pd.merge(frame1, frame2, on = 'check_number')

frame['sex'] = frame['sex'].map({'m':1,'M':1,'м':1,'М':1,'ж':0,'Ж':0})
frame = frame[frame['sex'].notna()]

frame['row_number'] = frame['row_number'].fillna(frame['row_number'].max())

frame['liters_drunk'] = frame['liters_drunk'].map(lambda x: None if x < 0 or x >= 7 else x)
frame['liters_drunk'] = frame['liters_drunk'].fillna(frame['liters_drunk'].mean(skipna = True))

print(frame)
