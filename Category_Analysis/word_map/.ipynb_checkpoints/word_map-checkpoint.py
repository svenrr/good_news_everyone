import numpy as np
import pandas as pd
import plotly.express as px


df = pd.read_csv('data.csv')
df.dropna(inplace=True)
df['word_length'] = [len(s) for s in df.word.values]

fig = px.scatter(df, x="x", y="y",
                 color="cluster",
                 text='word',
                 size='word_length',
                 size_max=50
)

fig.show()