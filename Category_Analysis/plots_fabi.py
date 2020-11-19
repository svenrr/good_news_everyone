import matplotlib.pyplot as plt
import numpy as np

def set_plot_stuff(plt, title, xlabel, ylabel):
    
    if title != None: plt.title(title)
    if xlabel != None: plt.xlabel(xlabel)
    if ylabel != None: plt.ylabel(ylabel)
        
    return

def set_axis_stuff(ax, title, xlabel, ylabel):
    
    if title != None: ax.set_title(title)
    if xlabel != None: ax.set_xlabel(xlabel)
    if ylabel != None: ax.set_ylabel(ylabel)

    return


def simple_bar(x, y, ax=None, title=None, xlabel=None, ylabel=None):
    
    if ax == None:
        plt.bar(x, y, edgecolor='black')
        set_plot_stuff(plt, title, xlabel, ylabel)

    else:
        ax.bar(x,y, edgecolor='black')
        set_axis_stuff(ax, title, xlabel, ylabel)
    
    return


def simple_h_bar(x, y, ax=None, title=None, xlabel=None, ylabel=None):
    
    if ax == None:
        plt.barh(x, y, edgecolor='black')
        set_plot_stuff(plt, title, xlabel, ylabel)

    else:
        ax.barh(x, y, edgecolor='black')
        set_axis_stuff(ax, title, xlabel, ylabel)
    
    return



def bar_adjacent_labels(x, y, ax=None, title=None, xlabel=None, ylabel=None):
    """
    plots a bar chart with multiple plots per category and strings as labels
    input y as dict
    """
    
    n_indices = len(x)
    ind = np.arange(n_indices)

    width = 0.35
    
    if ax == None:
        fig, ax = plt.subplots(figsize=(12,6))
        fig.tight_layout()

    #for y_ in y:
    ax.bar(ind - width/2, list(y.values())[0], width, label=list(y.keys())[0], edgecolor='black')
    ax.bar(ind + width/2, list(y.values())[1], width, label=list(y.keys())[1], edgecolor='black')
    ax.set_xticks(ind)
    ax.set_xticklabels(x)
    ax.legend()

    set_axis_stuff(ax, title, xlabel, ylabel)
    
    return




def few_points_line_plot(x, y, y2=None, ax=None, title=None, xlabel=None, ylabel=None):
    """
    Line plot with points for mutliple y values.
    Input y as dictionary, with keys as label names.
    If second line for every other is needed, put in as y2.
    """
    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.tight_layout()
        
    colors = ['blue', 'red', 'green', 'orange', 'brown']
    points = ['o', 'v', '>', 's', 'D', 'x']
    count=0
    for key in y.keys():
        ax.plot(x, y[key], c=colors[count], marker=points[count], label=key)
        if y2 != None:
            ax.plot(x, y2[key], c=colors[count], marker=points[count], linestyle='dashed')
        count += 1
    ax.legend();
    
    set_axis_stuff(ax, title, xlabel, ylabel)
    
    return