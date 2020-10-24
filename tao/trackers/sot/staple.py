import subprocess
import logging
import os
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.io import loadmat

from tao.utils.paths import ROOT_DIR
from .base import Tracker


STAPLE_ROOT = ROOT_DIR / 'third_party/staple'


class StapleTracker(Tracker):
    def init(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

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
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        region = [x0, y0, w, h]
        images_list = NamedTemporaryFile('w')
        images_list.writelines([f'{x}\n' for x in img_files])
        images_list.seek(0)
        # print(images_list.name)
        # print('hi')
        # subprocess.run(['cat', images_list.name], stderr=subprocess.STDOUT)
        # print('hi')
        # print([f'{x}\n' for x in img_files][:5])
        # print('hi')
        # print(img_files)
        # print('hi')

        output = NamedTemporaryFile('w', suffix='.mat')
        command = [
            'matlab', '-r',
            f"runTrackerTao('{images_list.name}', {region}, '{output.name}'); "
            f"quit"
        ]
        # Conda is clashing with MATLAB here, causing an error in C++ ABIs.
        # Unsetting LD_LIBRARY_PATH fixes this.
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = ''
        try:
            subprocess.check_output(command,
                                    stderr=subprocess.STDOUT,
                                    cwd=str(STAPLE_ROOT),
                                    env=env)
        except subprocess.CalledProcessError as e:
            logging.fatal('Failed command.\nException: %s\nOutput %s',
                          e.returncode, e.output.decode('utf-8'))
            raise

        result = loadmat(output.name)['results'].squeeze()
        images_list.close()
        output.close()

        boxes = result['res'].item()
        # width, height -> x1, y1
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        scores = result['scores'].item()
        boxes = np.hstack((boxes, scores))
        return boxes, None, None
