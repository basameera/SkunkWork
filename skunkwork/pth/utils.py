"""SkunkWork Pytorch Utils"""
from PIL import Image
import skunkwork
import warnings
from ..utils import getListOfFiles
import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

__all__ = ["getSplitByPercentage", "model_summary"]


def getSplitByPercentage(len=0, train_percentage=0.8, n_output=skunkwork.SPLIT_LEVEL_3):
    if train_percentage > 0.0 and train_percentage < 1.0:
        if n_output == 3:
            # train, val, test
            train_p = int(train_percentage*len)
            valid_p = (len - train_p)//2
            return [train_p, valid_p, len - train_p - valid_p]
        elif n_output == 2:
            # train, val
            train_p = int(train_percentage*len)
            valid_p = (len - train_p)
            return [train_p, valid_p]
        else:
            raise ValueError('n_output is either 2 or 3.')
    else:
        raise ValueError('Value should be between 0 and 1.')


def imgResize(source_folder='data', destination_folder='save', size=256, keep_aspect_ratio=True, use_og_name=True):
    fileList = getListOfFiles(source_folder)

    for id, path in enumerate(fileList):

        im = Image.open(path)
        width, height = im.size

        if isinstance(size, int):
            new_size = (size, size)
            if keep_aspect_ratio:
                if height < width:
                    new_size = ((size * width) // height, size)
                if height > width:
                    new_size = (size, (size * height) // width)
        if isinstance(size, tuple):
            new_size = size

        im = im.resize(new_size, Image.ANTIALIAS)
        fname = destination_folder+'/' + \
            str(id) + '.' + os.path.basename(path).split('.')[1]
        if use_og_name:
            fname = destination_folder+'/'+os.path.basename(path)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        print(id, ' | ', (width, height), ' -> ', new_size,
              ' | ', os.path.basename(path), '\t\t-> ', fname)
        im.save(fname)


def model_summary(model, *input_size, batch_size=-1, device="cuda", show=True):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]

            # only print nn modules
            if hasattr(nn, class_name):
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    if show:
        print("\n----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format(
            "Layer (type)", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0

    mdl_layers = []

    mdl_layers.append(('Input-0', '[{}, {}, {}, {}]'.format(
        batch_size, input_size[0][0], input_size[0][1], input_size[0][2]), '0'))
    if show:
        print("{:>20}  {:>25} {:>15}".format(
            'Input-0', '[{}, {}, {}, {}]'.format(batch_size, input_size[0][0],
                                               input_size[0][1], input_size[0][2]), '0')
              )
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )

        mdl_layers.append((layer, str(summary[layer]["output_shape"]), "{0:,}".format(
            summary[layer]["nb_params"])))

        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        if show:
            print(line_new)
    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) *
                           batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    if show:
        print("================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print(
            "Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------------------------------\n")

    mdl_summary = OrderedDict()
    mdl_summary['layers'] = mdl_layers
    mdl_summary['total_params'] = float(total_params.numpy())
    mdl_summary['trainable_params'] = float(trainable_params.numpy())

    mdl_summary['total_input_size_MB'] = float(total_input_size)
    mdl_summary['total_output_size_MB'] = float(total_output_size)
    mdl_summary['total_params_size_MB'] = float(total_params_size)

    return mdl_summary


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
