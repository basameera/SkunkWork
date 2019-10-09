from . import constants as c

from . import skunkwork
from .skunkwork import *

from . import utils
from .utils import *

from . import cv
from .cv import *

from . import linalg
from .linalg import *

from . import plot3D
from .plot3D import *

from .version import __version__

__all__ = []
__all__.extend(skunkwork.__all__)
__all__.extend(utils.__all__)
__all__.extend(cv.__all__)
__all__.extend(linalg.__all__)
__all__.extend(plot3D.__all__)

__version__ = __version__
__author__ = 'Sameera Sandaruwan'
__author_email__ = 'basameera@pm.me'


# constants
SPLIT_LEVEL_2 = c.SPLIT_LEVEL_2
SPLIT_LEVEL_3 = c.SPLIT_LEVEL_3
