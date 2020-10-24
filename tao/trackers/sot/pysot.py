import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from .base import Tracker


class PysotTracker(Tracker):
    def __init__(self, config_file, model_path):
        super().__init__()
        cfg.merge_from_file(config_file)
        model = ModelBuilder()
        model.load_state_dict(
            torch.load(model_path,
                       map_location=lambda storage, loc: storage.cpu()))
        model.eval().cuda()
        self.tracker = build_tracker(model)

    def init(self, image, box):
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        image = np.array(image)[:, :, [2, 1, 0]]  # RGB -> BGR
        self.tracker.init(image, (x0, y0, w, h))

    def update(self, image):
        image = np.array(image)[:, :, [2, 1, 0]]  # RGB -> BGR
        output = self.tracker.track(image)
        x0, y0, w, h = output['bbox']
        box = (x0, y0, x0 + w, y0 + h)
        return box, output['best_score'], {}
