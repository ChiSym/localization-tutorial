# %%

import os
os.environ["ANYWIDGET_HMR"] = "1"

import anywidget
import traitlets
import pathlib

def js(path):
    return pathlib.Path(__file__).parent.parent / 'js_dist' / path

# %%

class HistogramWidget(anywidget.AnyWidget):
    _esm = js("histogram.js")
    values = traitlets.List(traitlets.Int).tag(sync=True)
    def __init__(self, init):
        super().__init__()
        self.values = init
        
import random
widget = HistogramWidget([random.randint(0, 999) for _ in range(1000)])
widget

# widget.send('foo')

# %%
