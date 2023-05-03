""" Utility functions """

import os
from rail.estimation.algos import bpz_version

RAIL_BPZ_DIR = os.path.abspath(os.path.join(os.path.dirname(bpz_version.__file__), '..', '..', '..', '..'))
