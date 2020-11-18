"""
Convert json files for different publsihers to one csv file
In the same folder create different publishers as folders, then put all jsons in these folders!
"""
# Imports
import pandas as pd
import numpy as np
from glob import glob
import os

# Check if file is directory
check_dir = (lambda x: os.path.isdir(x))

# Get all directories
t = []
for i in glob('*'):
    if(check_dir(i)):
        t.append(i)

# Get all jsons in these directories
tuples = []
for j in t:
    for i in glob(j + '/*.json'):
        tuples.append((j,i))
            
# Read all jsons, put them together, add publisher
l = []
for t in tuples:
    
    publisher = t[0]
    file = t[1]

    df = pd.read_json(file, lines=True)
    df['publisher'] = publisher
    l.append(df)

# Add all data frames together
df = pd.concat(l, axis=0, ignore_index=True)

### Save as csv sheet
df.to_csv('news_actual.csv', index=False)