import os


MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULES_DIR)

INPUTS = os.path.join(PROJECT_DIR, 'inputs')
TS_DIR = os.path.join(INPUTS, 'time_series_20170614')

FODITS_FILE = os.path.join(INPUTS, 'Bernese', 'FODITS.L11')
# velocity files from Bernese
VEL_FILE_REF = os.path.join(INPUTS, 'Bernese', 'SKPOS_FO.VEL')
VEL_FILE_SOL = os.path.join(INPUTS, 'Bernese', 'ADN_S2.VEL')
# dates of discontinuities for custom modification
DST_FILE = os.path.join(INPUTS, 'other', '20170614_changed.dst')
# custom intervals for exclusion
EXCL_FILE = os.path.join(INPUTS, 'other', 'custom.excl')
# borders of european countries
BORDERS_SHP = os.path.join(INPUTS, 'other', 'borders', '_hranice.shp')
# fi, la of all stations for final plots
COORS_FILE = os.path.join(INPUTS, 'other', 'positions_cln.txt')
# history of imput files
HIST_FILE = os.path.join(INPUTS, 'other', 'history_in.tspro')
