"""SkunkWork Utils"""
import os
import datetime
import warnings
import subprocess
import math


def clog(*args, end='\n'):
    msg = '>>> '+str(datetime.datetime.now()).split('.')[0][2:] + ' :'
    for s in args:
        msg = msg + ' ' + str(s)
    print(msg, end=end)


def printLine(len, end='\n'):
    if isinstance(len, int):
        for _ in range(len):
            print('=', end='')
        print(end, end='')
    else:
        raise TypeError('Input should be an int value.')


def getMaxLen(input, output, prev_indent=0):
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
                getMaxLen(value, output,
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

    TODO: set max header-footer dash length to the length of the console
    """
    warnings.warn(
        'Set max header-footer dash length to the length of the console')
    prettyPrint(input, heading=heading)


def prettyPrint(input, heading='', prev_indent=0):

    if isinstance(input, dict):
        zzz = []
        getMaxLen(input, zzz)
        maxFooterLen = max(zzz)

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
            printLine(int(a/2), end='')
            if heading != '':
                print('', heading, '', end='')
            printLine(a - int(a/2))

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
                prettyPrint(value, prev_indent=prev_indent + maxKeyLen + 1)
            else:
                print('', value)

        # footer
        if prev_indent == 0:
            printLine(maxFooterLen + 2)

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


# run
if __name__ == '__main__':
    main()
