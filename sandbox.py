import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["PYPHI_WELCOME_OFF"] = "yes"

import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

import pyphi
pyphi.config.PROGRESS_BARS = True # may need to comment this one out for bigger networks, but this is fine for now
pyphi.config.VALIDATE_SUBSYSTEM_STATES = False

import itertools
import pickle

with open("regressionTest.pickle", "rb") as f:
    datalst = pickle.load(f)

    print(datalst[0][0])
    print(datalst[0][2])

    # for mc_lst in datalst[1]:
    #     print([mc.phi for mc in mc_lst])
