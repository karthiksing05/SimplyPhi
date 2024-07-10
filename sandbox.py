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
pyphi.config.USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE = False

import itertools
import pickle

with open("optim_nn_results.pickle", "rb") as f:
    datalst = pickle.load(f)

    print(datalst[0])

    for mc_lst in datalst[1]:
        print([mc.phi for mc in mc_lst])
