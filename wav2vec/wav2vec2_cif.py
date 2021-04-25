# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import Tensor

THRESHOLD = 0.999


def get_alphas(encoder_output):
    padding_mask = encoder_output['padding_mask']

    if 'encoded' in encoder_output.keys():
        alphas = encoder_output['encoded'][:, :, -1]
    else:
        alphas = encoder_output['encoder_out'][:, :, -1]

    alphas = torch.sigmoid(alphas)
    alphas = alphas * (~padding_mask).float()

    return alphas


def cif(encoder_output, alphas, threshold=THRESHOLD, log=False):
    if type(encoder_output) is Tensor:
        hidden = encoder_output
    elif 'encoded' in encoder_output.keys():
        hidden = encoder_output['encoded']
    else:
        hidden = encoder_output['encoder_out']

    device = hidden.device
    B, T, H = hidden.size()

    # loop varss
    integrate = torch.zeros([B], device=device)
    frame = torch.zeros([B, H], device=device)
    # intermediate vars along time
    list_fires = []
    list_frames = []

    for t in range(T):
        alpha = alphas[:, t]
        distribution_completion = torch.ones([B], device=device) - integrate

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(fire_place,
                                integrate - torch.ones([B], device=device),
                                integrate)
        cur = torch.where(fire_place,
                          distribution_completion,
                          alpha)
        remainds = alpha - cur

        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(fire_place[:, None].repeat(1, H),
                            remainds[:, None] * hidden[:, t, :],
                            frame)

        if log:
            print('t: {}\t{:.3f} -> {:.3f}|{:.3f} fire: {}'.format(
                t, integrate[log], cur[log], remainds[log], fire_place[log]))

    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)
    list_ls = []
    len_labels = torch.round(alphas.sum(-1)).int()
    max_label_len = len_labels.max()
    for b in range(B):
        fire = fires[b, :]
        l = torch.index_select(frames[b, :, :], 0, torch.where(fire >= threshold)[0])
        pad_l = torch.zeros([max_label_len - l.size(0), H], device=device)
        list_ls.append(torch.cat([l, pad_l], 0))

        if log:
            print(b, l.size(0))

    if log:
        print('fire:\n', fires[log])
        print('fire place:\n', torch.where(fires[log] >= threshold))

    return torch.stack(list_ls, 0)


def resize(alphas, target_lengths, noise=0.0, threshold=THRESHOLD):
    """
    alpha in thresh=1.0 | (0.0, +0.21)
    target_lengths: if None, apply round and resize, else apply scaling
    """
    device = alphas.device
    # sum
    _num = alphas.sum(-1)

    num = target_lengths.float()
    num = num + noise * torch.rand(alphas.size(0)).to(device)

    # scaling
    _alphas = alphas * (num / _num)[:, None].repeat(1, alphas.size(1))

    # rm attention value that exceeds threashold
    while len(torch.where(_alphas > threshold)[0]):
        print('fixing alpha')
        xs, ys = torch.where(_alphas > threshold)
        for x, y in zip(xs, ys):
            if _alphas[x][y] >= threshold:
                mask = _alphas[x].ne(0).float()
                mean = 0.5 * _alphas[x].sum() / mask.sum()
                _alphas[x] = _alphas[x] * 0.5 + mean * mask

    return _alphas, _num
