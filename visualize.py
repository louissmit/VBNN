__author__ = 'louis'
import numpy as np
import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler

from bokeh.plotting import figure, output_server, cursession, show
from bokeh.models import Range1d

from watchdog.events import FileSystemEventHandler

class PlotHandler(FileSystemEventHandler):
    def __init__(self, root, figures, types):
        super(PlotHandler, self).__init__()
        self.root = root
        self.figures = figures
        self.types = types

    def on_modified(self, event):
        print "updating!!"
        update_plots(self.root, self.figures, self.types)

def read_data(filename):
    with open(filename) as f:
        content = [float(x.strip()) for x in f.readlines()]
    return content

def get_data(root, id):
    filename = root + "/" + id
    return read_data(filename)

def get_fold_data(root, type, k):
    filename = root + "/"+type+"-fold=" + str(k)
    return read_data(filename)

def combine_fold_data(root, type, k):
    result = np.array(get_fold_data(root, type, 1))
    for i in range(2, k+1):
        result += np.array(get_fold_data(root, type, i))
    return result/k

def setup_plots(root, types):
    colors = ["green", "blue", 'red', 'yellow']
    figures = {}
    for type, ids in types.iteritems():
        p = figure(title=type,
                plot_width=400,
                plot_height=400,
                # x_range=[0, 100]
        )
        for i, id in enumerate(ids):
            y = get_data(root, id)
            # y = combine_fold_data(root, id, 10)
            x = np.arange(len(y))
            p.line(x, y,
                   legend=id,
                   x_axis_label='time',
                   y_axis_label=id,
                   color=colors[i],
                   name=id)
            # p.x_range.end = len(y)
        figures[type] = p
    show(p)
    return figures

def update_plots(root, figures, types):
    # for type, ids in types.iteritems():
    for type, fig in figures.iteritems():
        for id in types[type]:
            renderer = fig.select(dict(name=id))
            ds = renderer[0].data_source
            y = get_data(root, id)
            # if len(y) > fig.x_range.end-5:
            #     print "heeeeey"
            #     fig.x_range = Range1d(start=0, end=len(y)+20)
            ds.data["y"] = y
            ds.data["x"] = np.arange(len(y))
            ds._dirty = True
            cursession().store_objects(ds)

    cursession().publish()

# if __name__ == "__main__":
#     to_plot = ['lc', 'trainerr', 'deverr']
#     output_server("animated_line")
#     root = "vbperm"
#     figs = setup_plots(root, to_plot)
#     update_plots(root, figs)
if __name__ == "__main__":
    root = "vb6kgrowfast"
    # to_plot = ['trainerr', 'deverr']
    to_plot = {
        "error" :  ['trainerr', 'deverr'],
        "accuracy" :  ['trainacc', 'devacc'],
        "lc" : ['lc'],
        # "variance" : ['variance']
        "variance" : ['mean variance', 'min. variance', 'max. variance'],
        "means" : ['mean means', 'min. means', 'max. means'],
        # "grads" : ['vle grad', 'vlc grad', 'mle grad', 'mlc grad'],
        "normratio" : ['mu normratio', 'var normratio']
    }
    # to_plot = {
    #      "error" :  ['trainerr', 'deverr'],
    #     "accuracy" :  ['trainacc', 'devacc'],
    # }
    output_server("animated_line")

    figs = setup_plots(root, to_plot)
    while True:
        print "updatign!"
        update_plots(root, figs, to_plot)
        time.sleep(5)

    # event_handler = PlotHandler(root, figs, to_plot)
    # observer = Observer()
    # observer.schedule(event_handler, path=root, recursive=False)
    # observer.start()
    #
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     observer.stop()
    # observer.join()