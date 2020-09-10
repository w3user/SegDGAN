#!/usr/bin/env python
# encoding: utf-8
import SegDGAN


def build_model(name):
    if name == 'segdensean':
        return SegDGAN
    else:
        print("%s is not be defined !" % name)

