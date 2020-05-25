import itertools
import logging
from collections import defaultdict
from copy import deepcopy

import numpy as np

from .tao import Tao


class TaoResults(Tao):
    def __init__(self, tao_gt, results, max_dets=300):
        """Constructor for Tao results.
        Args:
            tao_gt (Tao or str): Tao class instance, or str containing path
                of annotation file)
            results (str or List[dict]): Contains path of result file or a
                list of dicts. Each dict should be an annotation containing
                a 'track_id.' The score for a track will be set as the average
                of the score of all annotations in the track. To use an
                alternative mechanism, the caller should pre-compute the track
                score and ensure all annotations in the track have the same
                score.
            max_dets (int): max number of detections per image. The official
                value of max_dets for Tao is 300.
        """
        if isinstance(tao_gt, Tao):
            self.dataset = deepcopy(tao_gt.dataset)
        elif isinstance(tao_gt, str):
            self.dataset = self._load_json(tao_gt)
        else:
            raise TypeError("Unsupported type {} of tao_gt.".format(
                type(tao_gt)))

        self.logger = logging.getLogger('tao.results')
        self.logger.info("Loading and preparing results.")

        if isinstance(results, str):
            result_anns = self._load_json(results)
        else:
            # This path way is provided for efficiency, in case the JSON was
            # already loaded by the caller.
            self.logger.warn(
                "Assuming user provided the results in correct format.")
            result_anns = results

        merge_map = Tao._construct_merge_map(self.dataset)
        for x in result_anns:
            if x['category_id'] in merge_map:
                x['category_id'] = merge_map[x['category_id']]

        assert isinstance(result_anns, list), "results is not a list."

        self.ensure_unique_track_ids(result_anns)

        if max_dets >= 0:
            # NOTE: We limit detections per _frame_, not per video.
            result_anns = self.limit_dets_per_image(result_anns, max_dets)

        tracks = {}  # Map track_id to track object
        if "bbox" in result_anns[0]:
            for id, ann in enumerate(result_anns):
                x1, y1, w, h = ann["bbox"]
                x2 = x1 + w
                y2 = y1 + h

                if "segmentation" not in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]

                track_id = ann['track_id']
                if track_id not in tracks:
                    tracks[track_id] = {
                        'id': track_id,
                        'video_id': ann['video_id'],
                        'category_id': ann['category_id']
                    }
                assert tracks[track_id]['category_id'] == ann['category_id'], (
                    f'Annotations for track {track_id} have multiple '
                    f'categories')
                ann["area"] = w * h
                ann["id"] = id + 1

        self.dataset["annotations"] = result_anns
        self.dataset["tracks"] = list(tracks.values())
        self._create_index()

        _required_average = False
        for track_id, track_anns in self.track_ann_map.items():
            scores = [float(x['score']) for x in track_anns]
            unique_scores = set(scores)
            if len(unique_scores) > 1:
                _required_average = True
                avg = np.mean(scores)
                self.tracks[track_id]['score'] = avg
                for x in track_anns:
                    x['score'] = avg
            elif len(unique_scores) == 1:
                self.tracks[track_id]['score'] = unique_scores.pop()
        if _required_average:
            self.logger.warn(
                'At least one track had annotations with different scores; '
                'using average of individual annotation scores as track '
                'scores.')

        img_ids_in_result = [ann["image_id"] for ann in result_anns]

        assert set(img_ids_in_result) == (
            set(img_ids_in_result) & set(self.get_img_ids())
        ), "Results do not correspond to current Tao set."

    def ensure_unique_track_ids(self, result_anns):
        track_id_videos = {}
        for ann in result_anns:
            t = ann['track_id']
            if t not in track_id_videos:
                track_id_videos[t] = ann['video_id']
            assert ann['video_id'] == track_id_videos[t], (
                f'Track id {t} appears in more than one video: '
                f'{track_id_videos[t]} and {ann["video_id"]}')

    def limit_dets_per_image(self, anns, max_dets):
        img_ann = defaultdict(list)
        for ann in anns:
            img_ann[ann["image_id"]].append(ann)

        for img_id, _anns in img_ann.items():
            if len(_anns) <= max_dets:
                continue
            _anns = sorted(_anns, key=lambda ann: ann["score"], reverse=True)
            img_ann[img_id] = _anns[:max_dets]

        return [ann for anns in img_ann.values() for ann in anns]

    def get_top_results(self, img_id, score_thrs):
        raise NotImplementedError(
            'Unclear if this should be per image or per video')
        # LVIS implementation below:
        # ann_ids = self.get_ann_ids(img_ids=[img_id])
        # anns = self.load_anns(ann_ids)
        # return list(filter(lambda ann: ann["score"] > score_thrs, anns))
