import datetime
import logging
from collections import OrderedDict
from collections import defaultdict

import itertools
import numpy as np
from numba import jit
from tqdm import tqdm

from .tao import Tao
from .results import TaoResults


@jit
def bb_intersect_union(d, g):
    """Compute intersection and union separately.
    Inspired by
    <https://github.com/cocodataset/cocoapi/blob/636becdc73d54283b3aac6d4ec363cffbb6f9b20/common/maskApi.c#L109>

    >>> bb_intersect_union([0, 0, 20, 20], [0, 0, 20, 20], False)
    (400, 400)
    >>> bb_intersect_union([0, 0, 20, 20], [0, 0, 10, 10], False)
    (100, 400)
    >>> bb_intersect_union([0, 0, 20, 20], [0, 0, 10, 10], True)
    (100, 400)
    >>> bb_intersect_union([0, 0, 20, 20], [0, 0, 30, 30], True)
    (400, 400)
    >>> bb_intersect_union([10, 20, 10, 10], [10, 20, 5, 5], True)
    (25, 100)
    """
    dx, dy, dw, dh = d
    gx, gy, gw, gh = g

    detection_area = dw * dh
    groundtruth_area = gw * gh

    intersection_left = max(dx, gx)
    intersection_right = min(dx + dw, gx + gw)
    intersection_top = max(dy, gy)
    intersection_bottom = min(dy + dh, gy + gh)

    w = max(intersection_right - intersection_left, 0)
    h = max(intersection_bottom - intersection_top, 0)

    intersect = w * h
    union = detection_area + groundtruth_area - intersect
    return intersect, union


def compute_imagenetvid_iou(dt_track, gt_track, threshold=0.5):
    """
    Args:
        dt_track (dict): Map image id to [x0, y0, w, h]
        gt_track (dict): Map image id to [x0, y0, w, h]
    """
    image_ids = set(gt_track.keys()) | set(dt_track.keys())
    num_matched = 0
    num_total = 0
    for image in image_ids:
        g = gt_track.get(image, None)
        d = dt_track.get(image, None)
        if d and g:
            intersect, union = bb_intersect_union(d, g)
            if intersect > threshold * union:
                num_matched += 1
            num_total += 1

        if d or g:
            num_total += 1
    return num_matched / num_total


def compute_track_box_iou(dt_track, gt_track):
    """
    Args:
        dt_track (dict): Map image id to [x0, y0, w, h]
        gt_track (dict): Map image id to [x0, y0, w, h]
    """
    # Modified from YTVIS evaluation:
    # https://github.com/youtubevos/cocoapi/blob/f3f9948a1f749fb95797f31a9616872811eea559/PythonAPI/pycocotools/ytvoseval.py#L200
    i = 0
    u = 0
    image_ids = set(gt_track.keys()) | set(dt_track.keys())
    for image in image_ids:
        g = gt_track.get(image, None)
        d = dt_track.get(image, None)
        if d and g:
            i_, u_ = bb_intersect_union(d, g)
            i += i_
            u += u_
        elif not d and g:
            u += g[2] * g[3]
        elif d and not g:
            u += d[2] * d[3]
    assert i <= u
    return i / u if u > 0 else 0


def compute_avg_track_iou(dt_track, gt_track):
    """
    Args:
        dt_track (dict): Map image id to [x0, y0, w, h]
        gt_track (dict): Map image id to [x0, y0, w, h]
    """
    # Modified from YTVIS evaluation:
    # https://github.com/youtubevos/cocoapi/blob/f3f9948a1f749fb95797f31a9616872811eea559/PythonAPI/pycocotools/ytvoseval.py#L200
    ious = []
    image_ids = set(gt_track.keys()) | set(dt_track.keys())
    for image in image_ids:
        g = gt_track.get(image, None)
        d = dt_track.get(image, None)
        if d and g:
            i_, u_ = bb_intersect_union(d, g)
            ious.append(i_ / u_ if u_ > 0 else 0)
        elif (not d and g) or (d and not g):
            ious.append(0)
    return np.mean(ious)


class TaoEval:
    def __init__(self,
                 tao_gt,
                 tao_dt,
                 logger=None,
                 iou_type="bbox",
                 iou_3d_type="3d_iou"):
        """Constructor for TaoEval.
        Args:
            tao_gt (Tao or str): Tao class instance, or str containing path
                of annotation file
            tao_dt (TaoResult or str or List[dict]): TaoResult instance of
                str containing path of result file, or list of dict
            iou_type (str): segm or bbox evaluation
            iou_3d_type (str): "3d_iou", "avg_iou" or "imagenetvid".
        """
        if not logger:
            self.logger = logging.getLogger('tao.eval')
        elif isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        else:
            self.logger = logger

        if iou_type not in ["bbox", "segm"]:
            raise ValueError("iou_type: {} is not supported.".format(iou_type))

        if isinstance(tao_gt, Tao):
            self.tao_gt = tao_gt
        elif isinstance(tao_gt, str):
            self.tao_gt = Tao(tao_gt)
        else:
            raise TypeError("Unsupported type {} of tao_gt.".format(tao_gt))

        if isinstance(tao_dt, TaoResults):
            self.tao_dt = tao_dt
        elif isinstance(tao_dt, (str, list)):
            self.tao_dt = TaoResults(self.tao_gt, tao_dt)
        else:
            raise TypeError("Unsupported type {} of tao_dt.".format(tao_dt))

        # per-video per-category evaluation results
        self.eval_vids = defaultdict(list)
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iou_type=iou_type, iou_3d_type=iou_3d_type)
        self.results = OrderedDict()
        self.ious = {}  # ious between all gts and dts

        self.params.vid_ids = sorted(self.tao_gt.get_vid_ids())
        self.params.cat_ids = sorted(self.tao_gt.get_cat_ids())

    def _to_mask(self, anns, tao):
        for ann in anns:
            rle = tao.ann_to_rle(ann)
            ann["segmentation"] = rle

    def _prepare(self):
        """Prepare self._gts and self._dts for evaluation based on params."""
        cat_ids = self.params.cat_ids if self.params.cat_ids else None

        gt_anns = self.tao_gt.load_anns(
            self.tao_gt.get_ann_ids(vid_ids=self.params.vid_ids,
                                    cat_ids=cat_ids))
        dt_anns = self.tao_dt.load_anns(
            self.tao_dt.get_ann_ids(vid_ids=self.params.vid_ids,
                                    cat_ids=cat_ids))
        if len(gt_anns) == 0:
            raise ValueError(
                'Found no groundtruth annotations for given params')
        if len(dt_anns) == 0:
            raise ValueError('Found no predicted annotations for given params')

        # convert ground truth to mask if iou_type == 'segm'
        if self.params.iou_type == "segm":
            self._to_mask(gt_anns, self.tao_gt)
            self._to_mask(dt_anns, self.tao_dt)

        # set ignore flag
        gts = self.tao_gt.group_ann_tracks(gt_anns)
        dts = self.tao_dt.group_ann_tracks(dt_anns)
        for gt in gts:
            if "ignore" not in gt:
                gt["ignore"] = 0

        for gt in gts:
            self._gts[gt["video_id"], gt["category_id"]].append(gt)

        # For federated dataset evaluation we will filter out all dt for an
        # video which belong to categories not present in gt and not present in
        # the negative list for an video. In other words detector is not
        # penalized for categories about which we don't have gt information
        # about their presence or absence in an video.
        vid_data = self.tao_gt.load_vids(ids=self.params.vid_ids)
        # per video map of categories not present in video
        vid_nl = {d["id"]: d["neg_category_ids"] for d in vid_data}
        # per video list of categories present in video
        vid_pl = defaultdict(set)
        for track in gts:
            vid_pl[track["video_id"]].add(track["category_id"])
        # per video map of categories which have missing gt. For these
        # categories we don't penalize the detector for flase positives.
        self.vid_nel = {
            d["id"]: d["not_exhaustive_category_ids"]
            for d in vid_data
        }

        for dt in dts:
            vid_id, cat_id = dt["video_id"], dt["category_id"]
            if (self.params.use_cats and cat_id not in vid_nl[vid_id]
                    and cat_id not in vid_pl[vid_id]):
                continue
            self._dts[vid_id, cat_id].append(dt)

        # self.freq_groups = self._prepare_freq_group()

    def _prepare_freq_group(self):
        raise NotImplementedError
        freq_groups = [[] for _ in self.params.vid_count_lbl]
        cat_data = self.tao_gt.load_cats(self.params.cat_ids)
        for idx, _cat_data in enumerate(cat_data):
            frequency = _cat_data["frequency"]
            freq_groups[self.params.vid_count_lbl.index(frequency)].append(idx)
        return freq_groups

    def evaluate(self, show_progress=False):
        """
        Run per video evaluation on given videos and store results
        (a list of dict) in self.eval_vids.
        """
        self.logger.info("Running per video evaluation.")
        self.logger.info("Evaluate annotation type *{}*".format(
            self.params.iou_type))

        self.params.vid_ids = list(np.unique(self.params.vid_ids))

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        self._prepare()

        self.ious = {(vid_id, cat_id): self.compute_iou(vid_id, cat_id)
                     for vid_id in tqdm(self.params.vid_ids,
                                        desc='Computing IoUs',
                                        disable=not show_progress)
                     for cat_id in cat_ids}

        # loop through videos, area range, max detection number
        self.eval_vids = {(v, c, a, t):
                          self.evaluate_vid(vid_id, cat_id, area_rng, time_rng)
                          for c, cat_id in enumerate(cat_ids)
                          for a, area_rng in enumerate(self.params.area_rng)
                          for t, time_rng in enumerate(self.params.time_rng)
                          for v, vid_id in enumerate(self.params.vid_ids)}

    def _get_gt_dt(self, vid_id, cat_id):
        """Create gt, dt which are list of tracks. If use_cats is true
        only tracks corresponding to tuple (vid_id, cat_id) will be
        used. Else, all tracks in video are used and cat_id is not used.

        Returns:
            gt (List[dict]): List of track objects
            dt (List[dict]): List of track objects

            where track objects are of the form:
              {'track_id': int, 'annotations': List[dict], 'video_id': int}
        """
        if self.params.use_cats:
            gt = self._gts[vid_id, cat_id]
            dt = self._dts[vid_id, cat_id]
        else:
            gt = [
                _track
                for _cat_id in self.params.cat_ids
                for _track in self._gts[vid_id, _cat_id]
            ]
            dt = [
                _track
                for _cat_id in self.params.cat_ids
                for _track in self._dts[vid_id, _cat_id]
            ]
        return gt, dt

    def compute_iou(self, vid_id, cat_id):
        gt, dt = self._get_gt_dt(vid_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return []

        # Sort detections in decreasing order of score.
        idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in idx]

        if self.params.iou_type == "segm":
            ann_type = "segmentation"
        elif self.params.iou_type == "bbox":
            ann_type = "bbox"
        else:
            raise ValueError("Unknown iou_type for iou computation.")
        gt = [{g['image_id']: g[ann_type]
               for g in gt_track['annotations']} for gt_track in gt]
        dt = [{d['image_id']: d[ann_type]
               for d in dt_track['annotations']} for dt_track in dt]

        ious = np.zeros([len(dt), len(gt)])
        for i, j in np.ndindex(ious.shape):
            if self.params.iou_3d_type == '3d_iou':
                ious[i, j] = compute_track_box_iou(dt[i], gt[j])
            elif self.params.iou_3d_type == 'avg_iou':
                ious[i, j] = compute_avg_track_iou(dt[i], gt[j])
            elif self.params.iou_3d_type == 'imagenetvid':
                ious[i, j] = compute_imagenetvid_iou(dt[i], gt[j])
        return ious

    def evaluate_vid(self, vid_id, cat_id, area_rng, time_rng):
        """Perform evaluation for single category and video."""
        gt, dt = self._get_gt_dt(vid_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return None

        # Add another filed _ignore to only consider anns based on area range.
        for g in gt:
            duration = len(g['annotations'])
            if (g["ignore"]
                    or (g["area"] < area_rng[0] or g["area"] > area_rng[1])
                    or (duration < time_rng[0] or duration > time_rng[1])):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # Sort gt ignore last
        gt_idx = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gt_idx]

        # Sort dt highest score first
        dt_idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dt_idx]

        # load computed ious
        ious = (
            self.ious[vid_id, cat_id][:, gt_idx]
            if len(self.ious[vid_id, cat_id]) > 0
            else self.ious[vid_id, cat_id]
        )

        num_thrs = len(self.params.iou_thrs)
        num_gt = len(gt)
        num_dt = len(dt)

        # Array to store the "id" of the matched dt/gt
        gt_m = np.zeros((num_thrs, num_gt)) - 1
        dt_m = np.zeros((num_thrs, num_dt)) - 1

        gt_ig = np.array([g["_ignore"] for g in gt])
        dt_ig = np.zeros((num_thrs, num_dt))

        for iou_thr_idx, iou_thr in enumerate(self.params.iou_thrs):
            if len(ious) == 0:
                break

            for dt_idx, _dt in enumerate(dt):
                iou = min([iou_thr, 1 - 1e-10])
                # information about best match so far (m=-1 -> unmatched)
                # store the gt_idx which matched for _dt
                m = -1
                for gt_idx, _ in enumerate(gt):
                    # if this gt already matched continue
                    if gt_m[iou_thr_idx, gt_idx] > 0:
                        continue
                    # if _dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gt_ig[m] == 0 and gt_ig[gt_idx] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dt_idx, gt_idx] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dt_idx, gt_idx]
                    m = gt_idx

                # No match found for _dt, go to next _dt
                if m == -1:
                    continue

                # if gt to ignore for some reason update dt_ig.
                # Should not be used in evaluation.
                dt_ig[iou_thr_idx, dt_idx] = gt_ig[m]
                # _dt match found, update gt_m, and dt_m with "id"
                dt_m[iou_thr_idx, dt_idx] = gt[m]["id"]
                gt_m[iou_thr_idx, m] = _dt["id"]

        # For Tao we will ignore any unmatched detection if that category was
        # not exhaustively annotated in gt.
        dt_ig_mask = [
            d["area"] < area_rng[0]
            or d["area"] > area_rng[1]
            or len(d["annotations"]) < time_rng[0]
            or len(d["annotations"]) > time_rng[1]
            or d["category_id"] in self.vid_nel[d["video_id"]]
            for d in dt
        ]
        dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
        dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt
        # Based on dt_ig_mask ignore any unmatched detection by updating dt_ig
        dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == -1, dt_ig_mask))
        # store results for given video and category
        return {
            "video_id": vid_id,
            "category_id": cat_id,
            "area_rng": area_rng,
            "time_rng": time_rng,
            "dt_ids": [d["id"] for d in dt],
            "gt_ids": [g["id"] for g in gt],
            "dt_matches": dt_m,
            "gt_matches": gt_m,
            "dt_scores": [d["score"] for d in dt],
            "gt_ignore": gt_ig,
            "dt_ignore": dt_ig,
        }

    def accumulate(self):
        """Accumulate per video evaluation results and store the result in
        self.eval.
        """
        self.logger.info("Accumulating evaluation results.")

        if not self.eval_vids:
            self.logger.warn("Please run evaluate first.")

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        num_thrs = len(self.params.iou_thrs)
        num_recalls = len(self.params.rec_thrs)
        num_cats = len(cat_ids)
        num_area_rngs = len(self.params.area_rng)
        num_time_rngs = len(self.params.time_rng)
        num_vids = len(self.params.vid_ids)

        # -1 for absent categories
        precision = -np.ones(
            (num_thrs, num_recalls, num_cats, num_area_rngs, num_time_rngs)
        )
        recall = -np.ones((num_thrs, num_cats, num_area_rngs, num_time_rngs))

        # Initialize dt_pointers
        dt_pointers = {}
        for cat_idx in range(num_cats):
            dt_pointers[cat_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[cat_idx][area_idx] = {}
                for time_idx in range(num_time_rngs):
                    dt_pointers[cat_idx][area_idx][time_idx] = {}

        # Per category evaluation
        for cat_idx, area_idx, time_idx in itertools.product(
                range(num_cats), range(num_area_rngs), range(num_time_rngs)):
            E = [
                self.eval_vids[vid_idx, cat_idx, area_idx, time_idx]
                for vid_idx in range(num_vids)
            ]
            # Remove elements which are None
            E = [e for e in E if e is not None]
            if len(E) == 0:
                continue

            # Append all scores: shape (N,)
            dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
            dt_ids = np.concatenate([e["dt_ids"] for e in E], axis=0)

            dt_idx = np.argsort(-dt_scores, kind="mergesort")
            dt_scores = dt_scores[dt_idx]
            dt_ids = dt_ids[dt_idx]

            dt_m = np.concatenate([e["dt_matches"] for e in E],
                                  axis=1)[:, dt_idx]
            dt_ig = np.concatenate([e["dt_ignore"] for e in E],
                                   axis=1)[:, dt_idx]

            gt_ig = np.concatenate([e["gt_ignore"] for e in E])
            # num gt anns to consider
            num_gt = np.count_nonzero(gt_ig == 0)

            if num_gt == 0:
                continue

            tps = np.logical_and(dt_m != -1, np.logical_not(dt_ig))
            fps = np.logical_and(dt_m == -1, np.logical_not(dt_ig))

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

            dt_pointers[cat_idx][area_idx][time_idx] = {
                "dt_ids": dt_ids,
                "tps": tps,
                "fps": fps,
            }

            for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                num_tp = len(tp)
                rc = tp / num_gt
                if num_tp:
                    recall[iou_thr_idx, cat_idx, area_idx, time_idx] = rc[-1]
                else:
                    recall[iou_thr_idx, cat_idx, area_idx, time_idx] = 0

                # np.spacing(1) ~= eps
                pr = tp / (fp + tp + np.spacing(1))
                pr = pr.tolist()

                # Replace each precision value with the maximum precision
                # value to the right of that recall level. This ensures
                # that the  calculated AP value will be less suspectable
                # to small variations in the ranking.
                for i in range(num_tp - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]

                rec_thrs_insert_idx = np.searchsorted(
                    rc, self.params.rec_thrs, side="left"
                )

                pr_at_recall = [0.0] * num_recalls

                try:
                    for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                        pr_at_recall[_idx] = pr[pr_idx]
                except:  # noqa: E722
                    pass
                precision[iou_thr_idx, :, cat_idx, area_idx, time_idx] = (
                    np.array(pr_at_recall))

        self.eval = {
            "params": self.params,
            "counts": [
                num_thrs, num_recalls, num_cats, num_area_rngs, num_time_rngs
            ],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "dt_pointers": dt_pointers,
        }

    def _summarize(self,
                   summary_type,
                   iou_thr=None,
                   area_rng="all",
                   time_rng="all",
                   freq_group_idx=None):
        aidx = [
            idx
            for idx, _area_rng in enumerate(self.params.area_rng_lbl)
            if _area_rng == area_rng
        ]
        time_idx = [
            idx
            for idx, _time_rng in enumerate(self.params.time_rng_lbl)
            if _time_rng == time_rng
        ]

        if summary_type == 'ap':
            s = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:
                s = s[:, :, self.freq_groups[freq_group_idx], aidx, time_idx]
            else:
                s = s[:, :, :, aidx, time_idx]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, :, aidx, time_idx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self):
        """Compute and display summary metrics for evaluation results."""
        if not self.eval:
            raise RuntimeError("Please run accumulate() first.")

        max_dets = self.params.max_dets

        self.results["AP"] = self._summarize('ap')
        self.results["AP50"] = self._summarize('ap', iou_thr=0.50)
        self.results["AP75"] = self._summarize('ap', iou_thr=0.75)

        for area_rng in ["small", "medium", "large"]:
            key = ("AP", "area", area_rng, max_dets)
            self.results[key] = self._summarize('ap', area_rng=area_rng)

        for time_rng in ["short", "medium", "long"]:
            key = ("AP", "time", time_rng, max_dets)
            self.results[key] = self._summarize('ap', time_rng=time_rng)

        # self.results["APr"]  = self._summarize('ap', freq_group_idx=0)
        # self.results["APc"]  = self._summarize('ap', freq_group_idx=1)
        # self.results["APf"]  = self._summarize('ap', freq_group_idx=2)

        key = "AR@{}".format(max_dets)
        self.results[key] = self._summarize('ar')

        for area_rng in ["small", "medium", "large"]:
            key = ("AR", "area", area_rng, max_dets)
            self.results[key] = self._summarize('ar', area_rng=area_rng)

        for time_rng in ["short", "medium", "long"]:
            key = ("AR", "time", time_rng, max_dets)
            self.results[key] = self._summarize('ar', time_rng=time_rng)

    def run(self, show_progress=False):
        """Wrapper function which calculates the results."""
        self.evaluate(show_progress=show_progress)
        self.accumulate()
        self.summarize()

    def print_results(self):
        template = (
            " {:<18} {}"
            " @[ IoU={:<9} | area={:>6s} | dur={:>6s} | maxDets={:>3d} "
            "catIds={:>3s}] ="
            " {:0.3f}"
        )

        for key, value in self.results.items():
            max_dets = self.params.max_dets
            if "AP" in key:
                title = "Average Precision"
                _type = "(AP)"
            else:
                title = "Average Recall"
                _type = "(AR)"

            area_rng = "all"
            time_rng = "all"
            if isinstance(key, tuple):
                subset_type, subset_rng, max_dets = key[1:]
                cat_group_name = "all"
                if subset_type == "time":
                    time_rng = subset_rng[0]
                elif subset_type == "area":
                    area_rng = subset_rng[0]
                else:
                    raise ValueError('This should not happen')

            if len(key) > 2 and key[2].isdigit():
                iou_thr = (float(key[2:]) / 100)
                iou = "{:0.2f}".format(iou_thr)
            else:
                iou = "{:0.2f}:{:0.2f}".format(
                    self.params.iou_thrs[0], self.params.iou_thrs[-1]
                )

            if len(key) > 2 and key[2] in ["r", "c", "f"]:
                cat_group_name = key[2]
            else:
                cat_group_name = "all"

            self.logger.info(
                template.format(title, _type, iou, area_rng, time_rng,
                                max_dets, cat_group_name, value))

    def get_results(self):
        if not self.results:
            self.logger.warn("results is empty. Call run().")
        return self.results


class Params:
    def __init__(self, iou_type, iou_3d_type='3d_iou'):
        """Params for Tao evaluation API."""
        self.vid_ids = []
        self.cat_ids = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        # self.iou_thrs = np.linspace(
        #     0.5, 0.95, np.round((0.95 - 0.5) / 0.05) + 1, endpoint=True
        # )
        self.iou_thrs = [0.5]
        self.rec_thrs = np.linspace(
            0.0, 1.00, np.round((1.00 - 0.0) / 0.01) + 1, endpoint=True
        )
        self.max_dets = 300
        self.area_rng = [
            [0 ** 2, 1e5 ** 2],
            [0 ** 2, 32 ** 2],
            [32 ** 2, 96 ** 2],
            [96 ** 2, 1e5 ** 2],
        ]
        self.area_rng_lbl = ["all", "small", "medium", "large"]
        self.time_rng = [[0, 1e5], [0, 3], [3, 10], [10, 1e5]]
        self.time_rng_lbl = ["all", "short", "medium", "long"]
        self.use_cats = 1
        # We bin categories in three bins based how many videos of the training
        # set the category is present in.
        # r: Rare    :  < 10
        # c: Common  : >= 10 and < 100
        # f: Frequent: >= 100
        self.vid_count_lbl = ["r", "c", "f"]
        self.iou_type = iou_type
        # 3D IoU type, can be one of:
        #   3d_iou:   \sum_t intersect(d_t, g_t) / \sum_t union(d_t, g_t)
        #   avg_iou:  \sum_t (intersect(d_t, g_t) / union(d_t, g_t))
        #   imagenetvid:  \sum_t (intersect(d_t, g_t) / union(d_t, g_t) > 0.5) / t
        self.iou_3d_type = iou_3d_type


if __name__ == "__main__":
    import doctest
    doctest.testmod()
