import skunkwork
from skunkwork.skunkwork import swTest
from skunkwork.utils import clog
from skunkwork.pth.utils import getSplitByPercentage
from skunkwork.pth.pth import pth_test_func
from skunkwork.pth.pytorchCustomDataset import testClass as tc1
from skunkwork.pth.swTrainer import testClass as tc2

if __name__ == "__main__":
    clog(skunkwork.__version__)
    swt = swTest()
    swt.compile()
    clog(getSplitByPercentage(1000))
    pth_test_func()

    t1 = tc1('pytorch custom dataset')
    t1.getName()

    t2 = tc2('Trainer')
    t2.getName()

    