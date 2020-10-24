import importlib

import numpy as np

from pytracking.features.net_wrappers import NetWithBackbone

from .base import Tracker


class PytrackingTracker(Tracker):
    """Wrapper around PyTracking Trackers.

    This re-implements some code from pytracking.evaluation.Tracker, removing
    parts of the code from that class that write to disk."""
    def __init__(self, tracker_name, tracker_param, model_path=None):
        super().__init__()

        tracker_module = importlib.import_module(
            'pytracking.tracker.{}'.format(tracker_name))
        self.tracker_class = tracker_module.get_tracker_class()
        self.params = self.get_parameters(tracker_name,
                                          tracker_param,
                                          model_path=model_path)

    def get_parameters(self, tracker_name, tracker_param, model_path=None):
        """Get parameters."""
        param_module = importlib.import_module(
            'pytracking.parameter.{}.{}'.format(tracker_name,
                                                tracker_param))
        params = param_module.parameters()
        if model_path is not None:
            params.net = NetWithBackbone(net_path=model_path,
                                         use_gpu=params.use_gpu)
        return params

    def init(self, image, box):
        self.tracker = self.tracker_class(self.params)
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        image = np.array(image)[:, :, [2, 1, 0]]  # RGB -> BGR
        self.tracker.initialize(image, {'init_bbox': [x0, y0, w, h]})

    def update(self, image):
        image = np.array(image)[:, :, [2, 1, 0]]  # RGB -> BGR
        output = self.tracker.track(image)
        x0, y0, w, h = output['target_bbox']
        box = (x0, y0, x0 + w, y0 + h)
        return box, self.tracker.debug_info['max_score'], {}
