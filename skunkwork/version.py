from . import version_revision as vr
MAJOR = 0
MINOR = 1
BUILD = 3
REVISION = vr.REVISION

VERSION = (MAJOR, MINOR, BUILD, REVISION)

__version__ = ".".join(map(str, VERSION))
