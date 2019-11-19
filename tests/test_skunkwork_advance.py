import skunkwork

from skunkwork.skunkwork import *
from skunkwork.utils import *
from skunkwork.cv import *

from skunkwork.pth.utils import getSplitByPercentage
from skunkwork.pth.pytorchCustomDataset import testClass as tc1
from skunkwork.pth.swTrainer import testClass as tc2


if __name__ == "__main__":
    clog('skunkwork version:', skunkwork.__version__)

    VERBOSE = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    for vb in VERBOSE:
        clog(vb, verbose=vb)

    swt = swTest()
    swt.run()
    clog(getSplitByPercentage(1000))

    t1 = tc1('pytorch custom dataset')
    t1.getName()

    t2 = tc2('Trainer')
    t2.getName()
