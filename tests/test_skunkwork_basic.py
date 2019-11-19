import skunkwork

from skunkwork.skunkwork import *
from skunkwork.utils import *
from skunkwork.cv import *


if __name__ == "__main__":
    clog('skunkwork version:', skunkwork.__version__)

    VERBOSE = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    for vb in VERBOSE:
        clog(vb, verbose=vb)

    swt = swTest()
    swt.run()

