"""
SkunkWork Utils
===============

"""
import os
import datetime
import warnings
import subprocess
import math
import argparse
import string
import json
import yaml
from blessings import Terminal
import numpy as np
from collections import OrderedDict

__all__ = ["IoU", "makedirs", "info", "json_read", "json_write", "yaml_read", "yaml_write", "simple_cmd_args",
           "clog", "pprint", "getListOfFiles", "convert_size", "getFolderSize"]


class tcolors:
    """
    Terminal Colors
    ---

    https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
    """
    FG_BLACK = '\033[90m'
    FG_RED = '\033[91m'
    FG_GREEN = '\033[92m'
    FG_YELLOW = '\033[93m'
    FG_BLUE = '\033[94m'
    FG_MAGENTA = '\033[95m'
    FG_CYAN = '\033[96m'
    FG_WHITE = '\033[97m'

    BG_BLACK = '\033[100m'
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'
    BG_MAGENTA = '\033[105m'
    BG_CYAN = '\033[106m'
    BG_WHITE = '\033[107m'


class tformat:
    """https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
    """
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


class vcolors:
    """
    Verbose Colors
    ---

    https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
    """
    INFO = tcolors.FG_GREEN
    WARNING = tcolors.FG_YELLOW
    ERROR = tcolors.FG_RED
    CRITICAL = tformat.BOLD + tformat.UNDERLINE + tcolors.FG_RED


def sample_function(a, b):
    """Sample Function

    BIG
    ---

    Addition of a and b

    Arguments:
        a {int} -- Description of a
        b {int} -- Description of b

    Returns:
        int -- Description of return
    """
    return a + b


def IoU(a, b):
    """Intersection-over-Union

    Arguments:
        a {numpy array} -- [description]
        b {numpy array} -- [description]

    Returns:
        float -- [description]
    """
    P, T = a.astype(np.int8), b.astype(np.int8)
    bw_and = np.bitwise_and(P, T)
    bw_xor = np.bitwise_xor(P, T)
    TP = np.sum(bw_and)
    FP_FN = np.sum(bw_xor)
    IoU = TP / (TP + FP_FN)
    return IoU


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def info(obj):
    print(type(obj))
    if isinstance(obj, np.ndarray):
        print(obj.shape, obj.dtype)


def yaml_write(filename, data):
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def yaml_read(filename):
    with open(filename) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def json_read(filename):
    with open(filename) as f_in:
        return(json.load(f_in))


def json_write(filename, data):
    with open(filename, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)


def _arg_reform(params):
    alphabet = list(string.ascii_lowercase)
    alphabet.remove('h')
    if len(params) > 26:
        raise ValueError(
            'Can\'t handle more than 25 args.'.format(len(alphabet)))
        return

    if isinstance(params, dict):
        arg_dict = OrderedDict()
        for i, (key, value) in enumerate(params.items()):
            arg_dict[alphabet[i]] = (
                key + ' (default: {})'.format(value), value)
        return arg_dict
    else:
        raise TypeError('params should be dict type')


def simple_cmd_args(cmd_params):
    """[summary]

    Usage:

        cmd_params = OrderedDict(test=1,
                        wait_length=0
                        )

        args = simple_cmd_args(cmd_params)
        test = int(args['test'])

    """
    params = _arg_reform(cmd_params)
    # check if params is dict
    if isinstance(params, dict):
        parser = argparse.ArgumentParser(
            description='*** Simple cmd args - by skunkwork ***')
        for key, value in params.items():
            parser.add_argument('-'+key, help=value[0], default=value[1])
        output = OrderedDict()

        for key1, key2 in OrderedDict(zip(cmd_params, sorted(parser.parse_args().__dict__))).items():
            output[key1] = parser.parse_args().__dict__[key2]
        return output
    else:
        raise TypeError('params should be dict')


def clog(*args, end='\n', verbose='DEBUG'):
    """[summary]

    Keyword Arguments:
        end {str} -- [description] (default: {'\\n'})
        verbose {str} -- [DEBUG, INFO, WARNING, ERROR, CRITICAL, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN] (default: {'DEBUG'})
    """
    if verbose == 'DEBUG':
        _print(*args, end=end)
    elif verbose == 'INFO':
        _info(*args, end=end)
    elif verbose == 'WARNING':
        _warn(*args, end=end)
    elif verbose == 'ERROR':
        _error(*args, end=end)
    elif verbose == 'CRITICAL':
        _critical(*args, end=end)

    elif verbose == 'RED':
        _red(*args, end=end)
    elif verbose == 'GREEN':
        _green(*args, end=end)
    elif verbose == 'YELLOW':
        _yellow(*args, end=end)
    elif verbose == 'BLUE':
        _blue(*args, end=end)
    elif verbose == 'MAGENTA':
        _magenta(*args, end=end)
    elif verbose == 'CYAN':
        _cyan(*args, end=end)


def _info(*args, end='\n'):
    _print(*args, end=end, header=vcolors.INFO, footer=tformat.ENDC)


def _warn(*args, end='\n'):
    _print(*args, end=end, header=vcolors.WARNING, footer=tformat.ENDC)


def _error(*args, end='\n'):
    _print(*args, end=end, header=vcolors.ERROR, footer=tformat.ENDC)


def _critical(*args, end='\n'):
    _print(*args, end=end, header=vcolors.CRITICAL, footer=tformat.ENDC)


def _red(*args, end='\n'):
    _print(*args, end=end, header=tcolors.FG_RED, footer=tformat.ENDC)


def _green(*args, end='\n'):
    _print(*args, end=end, header=tcolors.FG_GREEN, footer=tformat.ENDC)


def _yellow(*args, end='\n'):
    _print(*args, end=end, header=tcolors.FG_YELLOW, footer=tformat.ENDC)


def _blue(*args, end='\n'):
    _print(*args, end=end, header=tcolors.FG_BLUE, footer=tformat.ENDC)


def _magenta(*args, end='\n'):
    _print(*args, end=end, header=tcolors.FG_MAGENTA, footer=tformat.ENDC)


def _cyan(*args, end='\n'):
    _print(*args, end=end, header=tcolors.FG_CYAN, footer=tformat.ENDC)


def _print(*args, end='\n', header='', footer=''):
    msg = '>>> '+str(datetime.datetime.now()).split('.')[0][2:] + ' :'
    for s in args:
        msg = msg + ' ' + str(s)
    print(header + msg + footer, end=end)


def _printLine(len, symbol='=', end='\n'):
    if isinstance(len, int):
        for _ in range(len):
            print(symbol, end='')
        print(end, end='')
    else:
        raise TypeError('Input should be an int value.')


def _getMaxLen(input, output, prev_indent=0):
    if isinstance(input, dict):
        maxKeyLen = 0
        maxValLen = 0
        for key, value in input.items():
            if len(str(key)) > maxKeyLen:
                maxKeyLen = len(str(key))
            if len(str(value)) > maxValLen:
                maxValLen = len(str(value))

        rawValLen = 0
        # data
        for key, value in input.items():
            klen = len(str(key))

            if isinstance(value, dict):
                _getMaxLen(value, output,
                           prev_indent=prev_indent + maxKeyLen + 1)
            else:
                if len(str(value)) > rawValLen:
                    rawValLen = len(str(value))

        a = maxKeyLen + prev_indent + rawValLen
        output.append(a)

    else:
        raise TypeError('Input should be a Dictionary object.')


def pprint(input, heading=''):
    """SkunkWork Pretty Print 

    Arguments:
        input {dict}

    Keyword Arguments:
        heading {str} -- (default: {''})
    """
    _prettyPrint(input, heading=heading)


def _prettyPrint(input, heading='', prev_indent=0):
    terminal = Terminal()
    max_terminal_width = terminal.width
    if isinstance(input, dict):
        zzz = []
        _getMaxLen(input, zzz)
        maxFooterLen = min([max(zzz), max_terminal_width])
        if max(zzz) > max_terminal_width:
            maxFooterLen -= 2

        maxKeyLen = 0
        maxValLen = 0
        for key, value in input.items():
            if len(str(key)) > maxKeyLen:
                maxKeyLen = len(str(key))
            if len(str(value)) > maxValLen:
                maxValLen = len(str(value))

        a = maxFooterLen - len(str(heading))
        if heading == '':
            a += 2
        # header
        if prev_indent == 0:
            _printLine(int(a/2), end='')
            if heading != '':
                print('', heading, '', end='')
            _printLine(a - int(a/2))

        # data
        for key, value in input.items():
            for _ in range(prev_indent):
                print(' ', end='')
            print(str(key), end='')
            klen = len(str(key))
            for _ in range(maxKeyLen - klen):
                print(' ', end='')

            print(':', end='')

            if isinstance(value, dict):
                print('')
                _prettyPrint(value, prev_indent=prev_indent + maxKeyLen + 1)
            else:
                print('', value)

        # footer
        if prev_indent == 0:
            _printLine(maxFooterLen + 2, symbol='-')

    else:
        raise TypeError('Input should be a Dictionary object.')


def getListOfFiles(sourcePath, sort=False, ext=['.jpg', '.png']):
    """getListOfFiles

    Arguments:
        sourcePath {[type]} -- [description]

    Keyword Arguments:
        ext {list} -- [description] (default: {['.jpg', '.png']})
    """
    listOfFile = os.listdir(sourcePath)

    if sort:
        listOfFile = sorted(os.listdir(sourcePath))

    allFiles = list()
    # Iterate over a4ll the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(sourcePath, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if os.path.isfile(fullPath):
                _, f_ext = os.path.splitext(fullPath)
                if (f_ext.upper() in ext) or (f_ext.lower() in ext):
                    allFiles.append(fullPath)

    return allFiles


def convert_size(size_bytes):
    size_bytes = int(size_bytes)
    if size_bytes == 0:
        return size_bytes, "0B"
    size_name = ("B", "K", "M", "G", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 1)
    return size_bytes, "%s%s" % (s, size_name[i])


def getFolderSize(sourcePath, sort=True):
    """getFolderSize

    Arguments:
        sourcePath {[type]} -- [description]

    """

    listOfFile = os.listdir(sourcePath)

    if sort:
        listOfFile = sorted(os.listdir(sourcePath))

    allFiles = list()
    # Iterate over a4ll the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(sourcePath, entry)
        # If entry is a directory then get the list of files in this directory
        if not entry.startswith('.'):
            # size = subprocess.check_output(['du', '-sh', fullPath]).split()[0].decode('utf-8')
            size = subprocess.check_output(
                ['du', '-s', fullPath]).split()[0].decode('utf-8')
            size_in_bytes, size = convert_size(int(size)*1024)
            allFiles.append((size_in_bytes, size, fullPath))

    import heapq
    heapq.heapify(allFiles)
    out_list = list()
    while allFiles:
        pop = heapq.heappop(allFiles)
        out_list.append((pop[1], pop[2]))
    return out_list


def main():
    config = {
        "name": "Mnist_LeNet",
        "n_gpu": 1,
        "extra": [n for n in range(100)],

        "arch": {
            "type": "MnistModel",
            "args": {

            }
        },
        "data_loader": {
            "type": "MnistDataLoader",
            "args": {
                "data_dir": "data/",
                "batch_size": 64,
                "shuffle": True,
                "validation_split": 0.1,
                "num_workers": 2
            }
        },
        "optimizer": {
            "type": "Adam",
            "args": {
                "lr": 0.001,
                "weight_decay": 0,
                "amsgrad": True
            }
        },
        "loss": "nll_loss",
        "metrics": [
            "my_metric", "my_metric2"
        ],
        "lr_scheduler": {
            "type": "StepLR",
            "args": {
                "step_size": 50,
                "gamma": 0.1
            }
        },
        "trainer": {
            "epochs": 100,
            "save_dir": "saved/",
            "save_freq": 1,
            "verbosity": 2,
            "monitor": "min val_loss",
            "early_stop": 10,
            "tensorboardX": True,
        }
    }

    pprint(config, 'config')


# run
if __name__ == '__main__':
    main()
