"""
@author: Erik Andres Escayola

@title: Create charts of model results

@description:
  0. Define preamble (user)
  1. Import model estimation results (BEAR results)
  2. Reframe data for charts
  3. Generate charts and save in publication format
"""

# %% Preamble

# Import packages
# =============================================================================
import os
import pathlib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm

from functions import get_data as get
from functions import plot_IRF as irf
from functions import plot_FEVD as fevd
from functions import plot_HD as hd

# Define model specs
# =============================================================================
v = 3  # number of variables
h = 40 + 2  # h horizon +2
c = 4  # time/lw.bound/median/up.bound

vv = v + 2  # extra initial condition and constant
cv = 2  # contrib./value
s = 80  # full sample
p = 4  # p number of lags
t = s - p + 1  # t estimation sample +1

# Define result configs
# =============================================================================
res = ["IRF", "FEVD", "HD"]
res_bear = ["IRF", "FEVD", "hist decomp"]
cc = ["BR", "MX"]
country_long = ["Brazil", "Mexico"]
colors = ["g", "b"]  # number of cc = number of colors
palettes = ["Greens", "Blues"]  # number of cc = number of palettes

# Define paths
# =============================================================================
WDPATH = pathlib.Path("main.py").parent.resolve()
DATA = {}
OUTPUTS = {}

for i in cc:
    DATA[i] = os.path.join(WDPATH, "example", "bvar_"+i, "results")
    OUTPUTS[i] = os.path.join(WDPATH, "example", "bvar_"+i, "charts")


# %% Data

# Get data
# =============================================================================
raw, clean, label, output = get.getData(cc, DATA, res, res_bear, v, vv, t, cv,
                                        h, c)

# %% Charts

# Plot IRF, FEVD, HD
# =============================================================================
irf.plotIRF(cc, colors, v, h, output, label, OUTPUTS)
fevd.plotFEVD(cc, palettes, v, h, output, label, OUTPUTS)
hd.plotHD(cc, palettes, v, t, output, label, OUTPUTS)