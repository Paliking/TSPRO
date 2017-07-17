import pandas as pd

import velocities
import constants

DST_FILE = constants.DST_FILE
EXCL_FILE = constants.EXCL_FILE
TS_DIR = constants.TS_DIR
COORS_FILE = constants.COORS_FILE
SHP_FILE = constants.BORDERS_SHP

df = pd.read_csv('./inputs/other/Brano_velos.csv', index_col='stat')
df.columns = ['n', 'e', 'u']
df = df/1000
velos2plot = df.to_dict(orient='index')
fake_SDs = (df/1000000000).to_dict(orient='index')
velocities.plot_velocities(COORS_FILE, SHP_FILE, velos2plot, fake_SDs, ellipse=True)