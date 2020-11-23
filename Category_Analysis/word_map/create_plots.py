"""
            Plot History
"""


"""
            Settings
"""
INPUT_FOLDER = 'simulation_results'

#### Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from time import time
import pickle
# Add functions path
import sys
sys.path.append('../../Functions')
from datasets import load_stratified_dataset
from sklearn.cluster import KMeans

"""
            Load stuff
"""

Map_History = pickle.load(open(INPUT_FOLDER + '/Map_History.pkl', 'rb'))

words = pickle.load(open(INPUT_FOLDER + '/words.pkl', 'rb'))

likelihood_df = pickle.load(open(INPUT_FOLDER + "/likelihood_df.pkl", "rb"))

count_total = pickle.load(open(INPUT_FOLDER + "/count_total.pkl", "rb"))


"""
            Give Words Categories
"""

#Give word the category where it has most occurences.

#word_categories = [word_count.loc[word].argmax() for word in words]

word_categories = [likelihood_df.loc[word].argmax() if likelihood_df.loc[word].max() > 0 else 7 for word in words]
word_categories_names = [likelihood_df.columns[cat] for cat in word_categories]

"""
            Sizes of markers
"""

#Create scaler to get sizes of markers between a range

min_size, max_size = 30, 80

class Scaler():
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
        
    def fit(self, words, count_total):
        l = []
        for word in words:
            l.append(count_total[word])
        self.min_count = min(l)
        self.max_count = max(l)
        
    def transform(self, x):
        
        return (x - self.min_count)/(self.max_count - self.min_count) * (self.max_size - self.min_size) + self.min_size

scaler = Scaler(min_size, max_size)
scaler.fit(words, count_total)


"""
            Create gif
"""

if True:

    import imageio
    import io

    def plot_for_offset(Map):
        
        if False:
            #################  K Means  #################
            # Set number of clusters
            n_clusters = 7

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(Map)
            idx = kmeans.fit_predict(Map)
            kmeans.score(Map)
        else:
            ########### Categories by word count per category ############
            idx = word_categories
        
        #################  Plot  #########################

        fig, ax = plt.subplots(figsize=(12,12))
        fig.tight_layout()
        
        # Set range
        ax.set_xlim(-15000,15000)
        ax.set_ylim(-15000,15000)
        
        # List of colors
        colors = ['green', 'orange', 'red', 'blue', 'yellow', 'brown', 'violet', 'grey']

        # Plot all words
        for word, idx_, vec, label in zip(words, idx, Map, word_categories_names):
            x = vec[0]
            y = vec[1]
            ms = scaler.transform(count_total[word])
            ax.plot(x, y, marker='o', ms=ms, c=colors[idx_], alpha=0.7, linestyle='none')
            plt.annotate(word, (x, y), ha='center', va='center', size=10)

        # Create legend
        l = []
        for i in range(len(likelihood_df.columns)):
            l.append(mpatches.Patch(color=colors[i], label=likelihood_df.columns[i]))
        ax.legend(handles=l)

        # Used to return the plot as an image array
        plt.close(fig)
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')
        io_buf.seek(0)
        image = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()

        return image

    start_time = time()
    kwargs_write = {'fps':1.0, 'quantizer':'nq'}

    if True:
        map_hist_less = []
        count = 0
        for m in Map_History:
            if count%100 == 0:
                map_hist_less.append(m)
            count+=1
        imageio.mimsave('./map.gif', [plot_for_offset(m) for m in map_hist_less], fps=15)
    else:
        imageio.mimsave('./map.gif', [plot_for_offset(m) for m in Map_History], fps=30)
        
    print('Plotting took {:.2f}m'.format((time()-start_time)/60))

"""
            Create interactive plot
"""

# for output config: 
#import bokeh.io 
#bokeh.io.reset_output()
# shows output in notebook:
#bokeh.io.output_notebook()

from bokeh.models import ColumnDataSource, Label, LabelSet, Range1d, Legend, LegendItem
from bokeh.plotting import figure, output_file, show

output_file("map.html", title="Word Map")
Map = Map_History[-1]
c = {0:'blue', 1: 'red', 2: 'green', 3: 'brown', 4: 'yellow', 5: 'orange', 6: 'pink', 7: 'grey'}
color = [c[i] for i in word_categories]
radius = [scaler.transform(count_total[word])/5 for word in words]
#print(radius)

source = ColumnDataSource(data=dict(x=Map[:,0],
                                    y=Map[:,1],
                                    names=words,
                                    color=color,
                                    radius=radius,
                                    categories=word_categories_names))

p = figure(title='Word Map')#,
#           x_range=Range1d(140, 275))
p.scatter(x='x', y='y', size='radius', source=source, fill_color='color', color='color', legend='categories')
#p.xaxis[0].axis_label = 'Weight (lbs)'
#p.yaxis[0].axis_label = 'Height (in)'

labels = LabelSet(x='x', y='y', text='names', level='glyph',
                  text_font_size='10pt',
                  #x_offset=0, y_offset=0,
                  text_align='center',
                  source=source, render_mode='canvas')

#citation = Label(x=70, y=70, x_units='screen', y_units='screen',
#                 text='Done by Babo C. 2020-11-20', render_mode='css',
#                 border_line_color='black', border_line_alpha=1.0,
#                 background_fill_color='white', background_fill_alpha=1.0)

p.add_layout(labels)
#p.add_layout(citation)

show(p)