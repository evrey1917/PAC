import pandas as pd
import numpy as np

frame1 = pd.read_csv('cinema_sessions.csv', index_col=0)
frame2 = pd.read_csv('titanic_with_labels.csv', index_col=0)

frame = pd.merge(frame1, frame2, on = 'check_number')

frame = frame[frame['sex'].isin(['м','М','ж','Ж'])]
frame.loc[frame['sex'].isin(['ж','Ж']), 'sex'] = 0
frame.loc[frame['sex'].isin(['m','M','м','М']), 'sex'] = 1

frame['row_number'] = frame['row_number'].fillna(frame['row_number'].max())

frame.loc[frame['liters_drunk'] < 0, 'liters_drunk'] = None
frame.loc[frame['liters_drunk'] >= 7, 'liters_drunk'] = None
frame['liters_drunk'] = frame['liters_drunk'].fillna(frame['liters_drunk'].mean(skipna = True))

print(frame)