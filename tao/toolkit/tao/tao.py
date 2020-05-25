"""
API for accessing the Tao Dataset

Annotation file format:
{
    "info" : info,
    "images" : [image],
    "videos": [video],
    "tracks": [track],
    "annotations" : [annotation],
    "categories": [category],
    "licenses" : [license],
}

info: As in MS COCO
image: {
    "id" : int,
    "video_id": int,
    "file_name" : str,
    "license" : int,
    # Redundant fields for COCO-compatibility
    "width": int,
    "height": int,
    "frame_index": int
}
video: {
    "id": int,
    "name": str,
    "width" : int,
    "height" : int,
    "neg_category_ids": [int],
    "not_exhaustive_category_ids": [int]
}
track: {
    "id": int,
    "category_id": int,
    "video_id": int
}
category: {
    "id": int,
    "name": str,
    "synset": str,
    "supercategory": str
}
annotation: {
    "image_id": int,
    "track_id": int,
    "bbox": [x,y,width,height],
    "area": float,
    # Redundant field for compatibility with COCO scripts
    "category_id": int
}
license: {
    "id" : int,
    "name" : str,
    "url" : str,
}
"""

import json
import logging
from collections import defaultdict


class Tao:
    def __init__(self, annotation_path, logger=None):
        """Class for reading and visualizing annotations.
        Args:
            annotation_path (str): location of annotation file
        """
        if not logger:
            self.logger = logging.getLogger('tao.tao')
        elif isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        else:
            self.logger = logger

        self.logger.info("Loading annotations.")

        if isinstance(annotation_path, dict):
            for key in ('info', 'images', 'annotations', 'categories',
                        'videos', 'tracks'):
                assert key in annotation_path, (
                    f'Provided dictionary does not contain key {key}')
            self.dataset = annotation_path
        else:
            self.dataset = self._load_json(annotation_path)

        assert (type(self.dataset) == dict), (
            "Annotation file format {} not supported.".format(
                type(self.dataset)))
        self._create_index()

    @staticmethod
    def _construct_merge_map(dataset):
        merge_map = {}
        for category in dataset['categories']:
            if 'merged' in category:
                for to_merge in category['merged']:
                    merge_map[to_merge['id']] = category['id']
        if not merge_map:
            logging.error('Did not merge any categories.')
        return merge_map

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _create_index(self):
        self.logger.info("Creating index.")

        self.merge_categories = Tao._construct_merge_map(self.dataset)
        for x in self.dataset['annotations'] + self.dataset['tracks']:
            if x['category_id'] in self.merge_categories:
                x['category_id'] = self.merge_categories[x['category_id']]

        self.vids = {x['id']: x for x in self.dataset['videos']}
        self.tracks = {x['id']: x for x in self.dataset['tracks']}
        self.cats = {x['id']: x for x in self.dataset['categories']}

        self.imgs = {}
        self.vid_img_map = defaultdict(list)
        for image in self.dataset['images']:
            self.imgs[image['id']] = image
            self.vid_img_map[image['video_id']].append(image)

        self.vid_track_map = defaultdict(list)
        for track in self.tracks.values():
            self.vid_track_map[track['video_id']].append(track)

        self.anns = {}
        self.img_ann_map = defaultdict(list)
        self.cat_img_map = defaultdict(list)
        self.track_ann_map = defaultdict(list)
        negative_anns = []
        for ann in self.dataset["annotations"]:
            # The category id is redundant given the track id, but we still
            # require it for compatibility with COCO tools.
            ann['bbox'] = [float(x) for x in ann['bbox']]
            if (ann['bbox'][0] < 0 or ann['bbox'][1] < 0 or ann['bbox'][2] <= 0
                    or ann['bbox'][3] <= 0):
                negative_anns.append(ann['id'])
            assert 'category_id' in ann, (
                f'Category id missing in annotation: {ann}')
            assert (ann['category_id'] == self.tracks[ann['track_id']]
                    ['category_id'])
            self.track_ann_map[ann['track_id']].append(ann)
            self.img_ann_map[ann["image_id"]].append(ann)
            self.cat_img_map[ann["category_id"]].append(ann["image_id"])
            self.anns[ann["id"]] = ann
        if negative_anns:
            self.logger.warn(f'{len(negative_anns)} annotations had negative '
                             f'values in coordinates!')
            self.logger.debug('Annotation ids with negative values:\n' +
                              (','.join(map(str, negative_anns))))

        self.logger.info("Index created.")

    def get_track_ids(self,
                      vid_ids=None,
                      img_ids=None,
                      cat_ids=None,
                      area_rng=None):
        return list({
            self.anns[x]['track_id']
            for x in self.get_ann_ids(vid_ids, img_ids, cat_ids, area_rng)
        })

    def group_ann_tracks(self, anns):
        tracks = {}
        for ann in anns:
            tid = ann['track_id']
            if tid not in tracks:
                tracks[tid] = self.tracks[tid].copy()
                tracks[tid]['annotations'] = []
            tracks[tid]['annotations'].append(ann)

        for tid, track in tracks.items():
            track['annotations'] = sorted(
                track['annotations'],
                key=lambda x: self.imgs[x['image_id']]['frame_index'])
            # Compute average area
            track['area'] = (sum(x['area'] for x in track['annotations']) /
                             len(track['annotations']))
        return list(tracks.values())

    def get_single_object_init(self, track_id, init_type='first'):
        if init_type == 'first':
            return self.get_kth_annotation(track_id, k=0)
        elif init_type == 'biggest':
            return max(self.track_ann_map[track_id], key=lambda x: x['area'])
        else:
            raise NotImplementedError(f'Unsupported init type, {init_type}')

    def get_kth_annotation(self, track_id, k):
        """Return kth annotation for a track."""
        return sorted(self.track_ann_map[track_id],
                      key=lambda x: self.imgs[x['image_id']]['frame_index'])[k]

    def get_ann_ids(self,
                    vid_ids=None,
                    img_ids=None,
                    cat_ids=None,
                    area_rng=None):
        """Get ann ids that satisfy given filter conditions.

        Args:
            vid_ids (int array): get anns for given videos
            img_ids (int array): get anns for given imgs
            cat_ids (int array): get anns for given cats
            area_rng (float array): get anns for a given area range
                (e.g. [0, inf])

        Returns:
            ids (int array): integer array of ann ids
        """
        anns = []
        if vid_ids is not None:
            # Get all image ids for given video ids, and intersect with img_ids
            # if necessary.
            video_images = []
            for video_id in vid_ids:
                video_images.extend(
                    [x['id'] for x in self.vid_img_map[video_id]])
            if img_ids is None:
                img_ids = video_images
            img_ids = list(set(img_ids) & set(video_images))

        if img_ids is not None:
            for img_id in img_ids:
                anns.extend(self.img_ann_map[img_id])
        else:
            anns = self.dataset["annotations"]

        # return early if no more filtering required
        if cat_ids is None and area_rng is None:
            return [_ann["id"] for _ann in anns]

        cat_ids = set(cat_ids)

        if area_rng is None:
            area_rng = [0, float("inf")]

        ann_ids = [
            _ann["id"]
            for _ann in anns
            if _ann["category_id"] in cat_ids
            and _ann["area"] > area_rng[0]
            and _ann["area"] < area_rng[1]
        ]
        return ann_ids

    def get_cat_ids(self):
        """Get all category ids.

        Returns:
            ids (int array): integer array of category ids
        """
        return list(self.cats.keys())

    def get_vid_ids(self):
        """Get all vid ids.

        Returns:
            ids (int array): integer array of image ids
        """
        return list(self.vids.keys())

    def get_img_ids(self):
        """Get all img ids.

        Returns:
            ids (int array): integer array of image ids
        """
        return list(self.imgs.keys())

    def _load_helper(self, _dict, ids):
        if ids is None:
            return list(_dict.values())
        else:
            return [_dict[id] for id in ids]

    def load_anns(self, ids=None):
        """Load anns with the specified ids. If ids=None load all anns.

        Args:
            ids (int array): integer array of annotation ids

        Returns:
            anns (dict array) : loaded annotation objects
        """
        return self._load_helper(self.anns, ids)

    def load_tracks(self, ids=None):
        """Load tracks with the specified ids. If ids=None load all anns.

        Args:
            ids (int array): integer array of annotation ids

        Returns:
            tracks (dict array) : loaded track objects. Contains fields:
                'id', 'annotations', 'category_id'
        """
        return self._load_helper(self.tracks, ids)

    def load_cats(self, ids):
        """Load categories with the specified ids. If ids=None load all
        categories.

        Args:
            ids (int array): integer array of category ids

        Returns:
            cats (dict array) : loaded category dicts
        """
        return self._load_helper(self.cats, ids)

    def load_imgs(self, ids):
        """Load images with the specified ids. If ids=None load all images.

        Args:
            ids (int array): integer array of image ids

        Returns:
            imgs (dict array) : loaded image dicts
        """
        return self._load_helper(self.imgs, ids)

    def load_vids(self, ids):
        """Load videos with the specified ids. If ids=None load all videos.

        Args:
            ids (int array): integer array of video ids

        Returns:
            imgs (dict array) : loaded image dicts
        """
        return self._load_helper(self.vids, ids)
