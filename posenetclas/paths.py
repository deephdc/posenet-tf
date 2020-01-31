"""
Miscellaneous functions manage paths.

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""

import os.path
from datetime import datetime

from posenetclas import config


homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONF = config.conf_dict()
timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')


def get_base_dir():
    base_dir = CONF['general']['base_directory']
    if os.path.isabs(base_dir):
        return base_dir
    else:
        return os.path.abspath(os.path.join(homedir, base_dir))


def get_models_dir():
    return os.path.join(get_base_dir(), "models")


def get_dirs():
    return {'base dir': get_base_dir(),
            'models_dir': get_models_dir()
            }


def print_dirs():
    dirs = get_dirs()
    max_len = max([len(v) for v in dirs.keys()])
    for k,v in dirs.items():
        print('{k:{l:d}s} {v:3s}'.format(l=max_len + 5, v=v, k=k))


def main():
    return print_dirs()


if __name__ == "__main__":
    main()
