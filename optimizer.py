# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim

def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    # 过滤掉 requires_grad=False 的参数
    for g in parameters:
        g["params"] = [p for p in g["params"] if p.requires_grad]
    parameters = [g for g in parameters if len(g["params"]) > 0]

    lr = float(config.TRAIN.BASE_LR)
    wd = float(getattr(config.TRAIN, "WEIGHT_DECAY", 0.005))

    opt_lower = 'adamw'
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=0.9, nesterov=True,
                              lr=lr, weight_decay=wd)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999),
                                lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_lower}")

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
