import skunkwork
from skunkwork.skunkwork import swTest

if __name__ == "__main__":
    print(skunkwork.__version__)
    swt = swTest()
    swt.compile()