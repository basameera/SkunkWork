import skunkwork
from skunkwork.skunkwork import swTest
from skunkwork.utils import clog
from skunkwork.pth.utils import getSplitByPercentage
from skunkwork.pth.pth import pth_test_func

if __name__ == "__main__":
    clog(skunkwork.__version__)
    swt = swTest()
    swt.compile()
    clog(getSplitByPercentage(1000))
    pth_test_func()