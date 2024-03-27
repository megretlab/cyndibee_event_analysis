
def misc():

    def figure(figsize=None, *args, **kwargs):
        'Temporary workaround for traditional figure behaviour with the ipympl widget backend'
        fig = plt.original_figure(*args,**kwargs)
        if figsize:
            w, h =  figsize
        else:
            w, h = plt.rcParams['figure.figsize']
        fig.canvas.layout.height = str(h) + 'in'
        fig.canvas.layout.width = str(w) + 'in'
        return fig

    try:
        plt.original_figure
    except AttributeError:
        plt.original_figure = plt.figure
        plt.figure = figure

    plt.figure = plt.original_figure
    del plt.original_figure

    from IPython import get_ipython
    import matplotlib as mpl


    class InlinePlot(object): 
        def __init__(self): 
            self.ipython = get_ipython()

        def __enter__(self): 

            self.ipython.magic('matplotlib inline')
           # mpl.interactive(False)

        def __exit__(self, exc_type, exc_value, traceback): 
            plt.show()
            self.ipython.magic('matplotlib widget')
            #mpl.interactive(True)

    mpl.get_backend()