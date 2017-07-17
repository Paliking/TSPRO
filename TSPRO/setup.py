import sys
from cx_Freeze import setup, Executable
import scipy
import os


addtional_mods = ['numpy.core._methods', 'matplotlib.backends.backend_tkagg',
                  'tkinter.filedialog']

includefiles_list = ['inputs/']

scipy_path = os.path.dirname(scipy.__file__)
includefiles_list.append(scipy_path)

base = None
if sys.platform == 'win32':
    base = "Win32GUI"

setup(
    name="MyProgram",
    version="0.1",
    description="MyDescription",
    options = {'build_exe': {'includes': addtional_mods,
                             'include_files': includefiles_list}},
    executables=[Executable("TSPRO/TSPRO.py", base=base)],
)
