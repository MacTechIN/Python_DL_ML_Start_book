#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:03:33 2023

@author: sl
"""

    
# 데이터 로딩 

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns


#%%

# 데이터 로딩 

train = pd.read_csv("train.csv")
test = pd.read_sas("test.csv")
#%%