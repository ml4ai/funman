import json
from funman.parameter_space import ParameterSpace
from funman.search_utils import Box, Point, decode_labeled_object
from .box_plotter import BoxPlotter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from IPython.display import clear_output

import logging
l = logging.getLogger(__file__)
l.setLevel(logging.INFO)

def plot_parameter_space(ps: ParameterSpace, alpha: float = 0.2, clear = False):
    custom_lines = [
        Line2D([0], [0], color='g', lw=4, alpha=alpha),
        Line2D([0], [0], color='r', lw=4, alpha=alpha),
    ]
    plt.title("Parameter Space")
    plt.xlabel("beta_0")
    plt.ylabel("beta_1")
    plt.legend(custom_lines, ["true", "false"])
    for b1 in ps.true_boxes:
        BoxPlotter.plot2DBoxList(b1, color='g', alpha=alpha)
    for b1 in ps.false_boxes:
        BoxPlotter.plot2DBoxList(b1, color='r', alpha=alpha)
    # plt.show(block=True)
    if clear:
        clear_output(wait=True)

# TODO this behavior could be pulled into search_utils if we
# find reason to pause and restart a search
def plot_cached_search(search_path, alpha: float = 0.2):
    true_boxes = []
    false_boxes = []
    true_points = []
    false_points = []
    
    with open(search_path) as f:
        for line in f.readlines():
            if len(line) == 0:
                continue
            data = json.loads(line)
            ((inst, label), typ) = decode_labeled_object(data)
            if typ is Box:
                if label == "true":
                    true_boxes.append(inst)
                elif label == "false":
                    false_boxes.append(inst)
                else:
                    l.info(f"Skipping Box with label: {label}")
            elif typ is Point:
                if label == "true":
                    true_points.append(inst)
                elif label == "false":
                    false_points.append(inst)
                else:
                    l.info(f"Skipping Point with label: {label}")
            else:
                l.error(f"Skipping invalid object type: {typ}")
    plot_parameter_space(ParameterSpace(true_boxes, false_boxes), alpha=alpha)