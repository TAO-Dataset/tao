"""Evaluate tao results (helper script)."""

import argparse
import logging
from pathlib import Path

from script_utils.common import common_setup
from tao.utils.evaluation import get_cfg_defaults, evaluate, log_eval
from tao.utils.yacs_util import merge_from_file_with_base


CONFIG_DIR = Path(__file__).resolve().parent / 'configs'


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--config',
                        type=Path,
                        default=CONFIG_DIR / 'default.yaml')
    parser.add_argument('--config-updates', nargs='*')

    args = parser.parse_args()

    if args.output_dir:
        tensorboard_dir = args.output_dir / 'tensorboard'
        if tensorboard_dir.exists():
            raise ValueError(
                f'Tensorboard dir already exists, not evaluating.')
        args.output_dir.mkdir(exist_ok=True, parents=True)
        log_path = common_setup(__file__, args.output_dir, args).name
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                            datefmt='%H:%M:%S')
        logging.info('Args:\n%s', vars(args))
        log_path = None

    cfg = get_cfg_defaults()
    merge_from_file_with_base(cfg, args.config)
    if args.config_updates:
        cfg.merge_from_list(args.config_updates)
    cfg.freeze()

    if args.output_dir:
        with open(args.output_dir / 'config.yaml', 'w') as f:
            f.write(cfg.dump())

    tao_eval = evaluate(args.annotations, args.predictions, cfg)
    log_eval(tao_eval, cfg, output_dir=args.output_dir, log_path=log_path)


if __name__ == "__main__":
    main()
