import logging
import os
import yaml
from typing import Any, Dict

from yacs.config import CfgNode
from yacs.config import _valid_type, _VALID_TYPES


BASE_KEY = "_BASE_"


def _load_yaml_with_base(filename: str, allow_unsafe: bool = False) -> CfgNode:
    """
    Just like `yaml.load(open(filename))`, but inherit attributes from its
        `_BASE_`.

    Modified from
    https://github.com/facebookresearch/fvcore/blob/99cb965c67e675dc3259cd490c1dd78ab03a55ff/fvcore/common/config.py

    Args:
        filename (str): the file name of the current config. Will be used to
            find the base config file.
        allow_unsafe (bool): whether to allow loading the config file with
            `yaml.unsafe_load`.
    Returns:
        (dict): the loaded yaml
    """
    with open(filename, "r") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            if not allow_unsafe:
                raise
            logger = logging.getLogger(__name__)
            logger.warning(
                "Loading config {} with yaml.unsafe_load. Your machine may "
                "be at risk if the file contains malicious content.".format(
                    filename
                )
            )
            f.close()
            with open(filename, "r") as f:
                cfg = yaml.unsafe_load(f)  # pyre-ignore

    if cfg is None:
        return cfg

    # pyre-ignore
    def merge_a_into_b(a: Dict[Any, Any], b: Dict[Any, Any]) -> None:
        # merge dict a into dict b. values in a will overwrite b.
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                assert isinstance(
                    b[k], dict), "Cannot inherit key '{}' from base!".format(k)
                merge_a_into_b(v, b[k])
            else:
                b[k] = v

    if BASE_KEY in cfg:
        base_cfg_file = cfg[BASE_KEY]
        if base_cfg_file.startswith("~"):
            base_cfg_file = os.path.expanduser(base_cfg_file)
        if not any(map(base_cfg_file.startswith,
                       ["/", "https://", "http://"])):
            # the path to base cfg is relative to the config file itself.
            base_cfg_file = os.path.join(os.path.dirname(filename),
                                         base_cfg_file)
        base_cfg = _load_yaml_with_base(base_cfg_file,
                                        allow_unsafe=allow_unsafe)
        del cfg[BASE_KEY]
        if base_cfg is None:
            return cfg

        merge_a_into_b(cfg, base_cfg)  # pyre-ignore
        return base_cfg
    return cfg


def merge_from_file_with_base(cfg,
                              cfg_filename: str,
                              allow_unsafe: bool = False) -> None:
    """
    Merge configs from a given yaml file.
    
    Modified from
    https://github.com/facebookresearch/fvcore/blob/99cb965c67e675dc3259cd490c1dd78ab03a55ff/fvcore/common/config.py

    Args:
        cfg_filename: the file name of the yaml config.
        allow_unsafe: whether to allow loading the config file with
            `yaml.unsafe_load`.
    """
    loaded_cfg = _load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
    loaded_cfg = type(cfg)(loaded_cfg)
    cfg.merge_from_other_cfg(loaded_cfg)


def cfg_to_dict(cfg_node, key_list=[]):
    if not isinstance(cfg_node, CfgNode):
        assert _valid_type(cfg_node), (
            "Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES))
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_to_dict(v, key_list + [k])
        return cfg_dict


def cfg_to_flat_dict(cfg_node, key_list=[]):
    if not isinstance(cfg_node, CfgNode):
        assert _valid_type(cfg_node), (
            "Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES))
        return cfg_node
    else:
        cfg_dict_flat = {}
        for k, v in dict(cfg_node).items():
            updated = cfg_to_dict(v, key_list + [k])
            if isinstance(updated, dict):
                for k1, v1 in updated.items():
                    cfg_dict_flat['.'.join(key_list + [k, k1])] = v1
            else:
                cfg_dict_flat['.'.join(key_list + [k])] = updated
        return cfg_dict_flat
