import numpy as np

from sort import associate_detections_to_trackers, KalmanBoxTracker


class SortWithDetectionId(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
    Sets key parameters for SORT
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.iou_threshold = iou_threshold

    def update(self, dets):
        """
        Args:
            dets (np.array): Shape (num_boxes, 5), where each row contains
                [x1, y1, x2, y2, score]

        Retruns:
            tracks (np.array): Shape (num_boxes, 6), where each row contains
                [x1, y1, x2, y2, detection_index, track_id]
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if (np.any(np.isnan(pos))):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = (
            associate_detections_to_trackers(dets, trks, self.iou_threshold))

        # update matched trackers with assigned detections
        track_to_det_index = {t: d for d, t in matched}
        # matched[i, 0] is matched to matched[i, 1]
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = track_to_det_index[t]
                trk.update(dets[d, :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
            track_to_det_index[len(self.trackers) - 1] = i
        i = len(self.trackers)
        for t, trk in reversed(list(enumerate(self.trackers))):
            d = trk.get_state()[0]
            det_id = track_to_det_index.get(t, -1)
            if ((trk.time_since_update < 1)
                    and (trk.hit_streak >= self.min_hits
                         or self.frame_count <= self.min_hits)):
                ret.append(
                    np.concatenate((d, [det_id, trk.id + 1])).reshape(
                        1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 6))
