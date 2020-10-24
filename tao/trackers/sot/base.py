import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tqdm
from PIL import Image

from tao.utils import vis

_GREEN = (18, 127, 15)
_GRAY = (218, 227, 218)
_BLACK = (0, 0, 0)
COLOR_BOX = COLOR_MASK = [255*x for x in (0.000, 0.447, 0.741)]
COLOR_TEXT = _GRAY
COLOR_TEXT_INACTIVE = _BLACK
COLOR_MASK_INACTIVE = COLOR_BOX_INACTIVE = _GRAY

WIDTH_BOX = 10
WIDTH_BOX_INACTIVE = 1

WIDTH_MASK = 2
BORDER_ALPHA_MASK = 0.9
WIDTH_MASK_INACTIVE = 1


class Tracker(ABC):
    @property
    def stateless(self):
        return False

    @abstractmethod
    def init(self, image, box):
        """
        Args:
            image (np.array): Shape (height, width, num_channels). RGB image.
            box (list of int): (x0, y0, x1, y1). 0-indexed coordinates from
                top-left.
        """
        pass

    @abstractmethod
    def update(self, image):
        """
        Args:
            image (np.array): Shape (height, width, num_channels). RGB image.

        Returns:
            box (list of int): (x0, y0, x1, y1). 0-indexed coordinates from
                top-left.
            score (float)
        """
        pass

    def track_yield(self,
                    img_files,
                    box,
                    yield_image=False,
                    **unused_extra_args):
        """
        Args:
            img_files (list of str/Path): Ordered list of image paths
            box (list of int): (x0, y0, x1, y1). 0-indexed coordinates from
                top-left.
            yield_image (bool): Whether to yield the original image. Useful
                if the caller wants to operate on images without having to
                re-read them from disk.

        Yields:
            box (np.array): Shape (5, ), containing (x0, y0, x1, y1, score).
                0-indexed coordinates from top-left.
            tracker_time (float): Time elapsed in tracker.
            image (optional, np.array): Image loaded from img_files; see
                yield_image.
        """
        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            if not image.mode == 'RGB':
                image = image.convert('RGB')
            image = np.array(image)

            start_time = time.time()
            if f == 0:
                self.init(image, box)
                elapsed_time = time.time() - start_time
                box = np.array([box[0], box[1], box[2], box[3], float('inf')])
                extra_output = {}
            else:
                output = self.update(image)
                assert len(output) in (2, 3)
                box, score = output[:2]
                extra_output = output[2] if len(output) == 3 else {}
                elapsed_time = time.time() - start_time
                box = np.array([box[0], box[1], box[2], box[3], score])
            if yield_image:
                yield box, elapsed_time, extra_output, image
            else:
                yield box, elapsed_time, extra_output

    @contextmanager
    def videowriter(self,
                    output_video,
                    width,
                    height,
                    fps=30,
                    ffmpeg_params=None):
        if isinstance(output_video, Path):
            output_video = str(output_video)
        if ffmpeg_params is None:
            ffmpeg_params = [
                '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2", '-pix_fmt',
                'yuv420p'
            ]
        with FFMPEG_VideoWriter(
                output_video,
                size=(width, height),
                fps=fps,
                ffmpeg_params=ffmpeg_params) as writer:
            yield writer

    def vis_single_prediction(self,
                              image,
                              box,
                              mask=None,
                              label=None,
                              mask_border_width=WIDTH_MASK,
                              mask_border_alpha=BORDER_ALPHA_MASK,
                              box_color=COLOR_BOX,
                              text_color=COLOR_TEXT,
                              mask_color=COLOR_MASK):
        """
        Args:
            image (np.array)
            box (list-like): x0, y0, x1, y1, score
            mask (np.array): Shape (height, width)
        """
        if mask is None:
            image = vis.vis_bbox(
                image, (box[0], box[1], box[2] - box[0], box[3] - box[1]),
                fill_color=box_color)
            if label is None:
                text = f'Object: {box[4]:.02f}'
            else:
                # text = f'{label}: {box[4]:.02f}'
                text = f'{label}'
            image = vis.vis_class(image, (box[0], box[1] - 2),
                                  text,
                                  font_scale=0.75,
                                  text_color=text_color)
        # if box[4] < 0.8:  # Draw gray masks when below threshold.
        #     mask_color = [100, 100, 100]
        if mask is not None:
            image = vis.vis_mask(
                image,
                mask,
                mask_color,
                border_thick=mask_border_width,
                border_alpha=mask_border_alpha)
        return image

    def vis_image(self,
                  image,
                  box,
                  mask=None,
                  label=None,
                  other_boxes=[],
                  other_masks=[],
                  other_labels=[],
                  vis_threshold=0.1):
        """
        Args:
            image (np.array)
            box (list-like): x0, y0, x1, y1, score
            mask (np.array): Shape (height, width)
            other_boxes (list[list-like]): Contains alternative boxes that
                were not selected.
            other_masks (list[list-like]): Contains masks for alternative
                boxes that were not selected.
        """
        return self.vis_single_prediction(image, box, mask, label=label)

    def track(self,
              img_files,
              box,
              show_progress=False,
              output_video=None,
              output_video_fps=30,
              visualize_subsample=1,
              visualize_threshold=0.1,
              return_masks=False,
              **tracker_args):
        """
        Like self.track, but collect all tracking results in numpy arrays.

        Args:
            img_files (list of str/Path): Ordered list of image paths
            box (list of int): (x0, y0, x1, y1). 0-indexed coordinates from
                top-left.
            output_vis
            return_masks (bool): If false, don't return masks. This is helpful
                for OxUvA, where collecting all the masks may use too much
                memory.

        Returns:
            boxes (np.array): Shape (num_frames, 5), contains
                (x0, y0, x1, y1, score) for each frame. 0-indexed coordinates
                from top-left.
            times (np.array): Shape (num_frames,), contains timings for each
                frame.
        """
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 5))
        if return_masks:
            masks = [None] * frame_num
        times = np.zeros(frame_num)

        pbar = partial(tqdm, total=len(img_files), disable=not show_progress)
        if output_video is None:
            for f, (box, elapsed_time, extra) in enumerate(
                    pbar(self.track_yield(img_files, box, **tracker_args))):
                boxes[f] = box
                times[f] = elapsed_time
                if return_masks:
                    masks[f] = extra.get('mask', None)
        else:
            output_video = Path(output_video)
            output_video.parent.mkdir(exist_ok=True, parents=True)
            # Some videos don't play in Firefox and QuickTime if '-pix_fmt
            # yuv420p' is not specified, and '-pix_fmt yuv420p' requires that
            # the dimensions be even, so we need the '-vf scale=...' filter.
            width, height = Image.open(img_files[0]).size
            with self.videowriter(
                    output_video, width=width, height=height,
                    fps=output_video_fps) as writer:
                track_outputs = self.track_yield(
                    img_files, box, yield_image=True, **tracker_args)
                for f, (box, elapsed_time, extra, image) in enumerate(
                        pbar(track_outputs)):
                    mask = extra.get('mask', None)
                    if mask is not None and mask.shape != image.shape[:2]:
                        logging.warn(
                            f'Resizing mask (shape {mask.shape}) to match '
                            f'image (shape {image.shape[:2]})')
                        new_h, new_w = image.shape[:2]
                        mask = np.asarray(
                            Image.fromarray(mask).resize(
                                (new_w, new_h), resample=Image.NEAREST))
                    other_boxes = extra.get('other_boxes', [])
                    other_masks = extra.get('other_masks', [])
                    label = extra.get('label', None)
                    other_labels = extra.get('other_labels', [])
                    if (f % visualize_subsample) == 0:
                        writer.write_frame(
                            self.vis_image(image,
                                           box,
                                           mask,
                                           label=label,
                                           other_boxes=other_boxes,
                                           other_masks=other_masks,
                                           other_labels=other_labels,
                                           vis_threshold=visualize_threshold))
                    boxes[f] = box
                    times[f] = elapsed_time
                    if return_masks:
                        masks[f] = mask
        if return_masks:
            return boxes, masks, times
        else:
            return boxes, None, times
