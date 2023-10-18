"""Config management."""


import argparse
from pathlib import Path
import inspect
from datetime import datetime, timezone

from hydra.experimental import initialize_config_dir, compose
from omegaconf import OmegaConf, DictConfig


def init_hydra_cfg() -> DictConfig:
    """Initialize Hydra config."""

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default=None, help='configuration file')
    parser.add_argument('--override', type=str, nargs='+', default=[], help='configuration override(s)')
    args = parser.parse_args()

    config_file = args.config
    config_file = Path(config_file)

    override = args.override

    config_name = config_file.stem
    config_dir = config_file.absolute().parent

    # Called by
    stack = inspect.stack()
    if len(stack) >= 2:
        frame = stack[1]
        called_by = frame.filename
    else:
        called_by = 'undef'

    called_by = Path(called_by)

    initialize_config_dir(config_dir=str(config_dir), job_name=str(called_by.stem))
    cfg = compose(config_name=config_name, overrides=override)

    now = datetime.now(timezone.utc).astimezone()
    date_str = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z%z')

    # DictConfig of struct mode doesn't accept insertion of new keys.
    # Dirty solution
    OmegaConf.set_struct(cfg, False)
    cfg._run = {
        "name": called_by.stem,
        "path": str(called_by.absolute()),
        "timestamp": date_str,
    }
    OmegaConf.set_struct(cfg, True)

    return cfg
