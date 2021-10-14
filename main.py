import os
import logging
import toml
import data_generator.source_data as sd
from inspect import getmembers, isfunction
from logging import config


log_cfg = toml.load(os.path.join(os.path.dirname(__file__), 'pyproject.toml'))
config.dictConfig(log_cfg)
_logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # print(dir(sd))
    print(getmembers(sd, isfunction))
