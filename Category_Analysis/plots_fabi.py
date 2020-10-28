def simple_bar(x, y, ax=None, title=None, xlabel=None, ylabel=None):
    
    if ax == None:
        import matplotlib.pyplot as plt
        plt.bar(x, y)
        set_plot_stuff(plt, title, xlabel, ylabel)

    else:
        ax.bar(x,y)
        set_axis_stuff(ax, title, xlabel, ylabel)
    
    return

def simple_h_bar(x, y, ax=None, title=None, xlabel=None, ylabel=None):
    
    if ax == None:
        import matplotlib.pyplot as plt
        plt.barh(x, y)
        set_plot_stuff(plt, title, xlabel, ylabel)

    else:
        ax.barh(x,y)
        set_axis_stuff(ax, title, xlabel, ylabel)
    
    return

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