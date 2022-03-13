"""
@author: Erik Andres Escayola

@title: Data analysis of model variables

@description:
  1. Import time-series data
  2. Preliminary analysis through plots and desriptive stats
  3. Compute tests on unit roots, lag-order, residual autocorrelation
  4. Test for Granger causality of all variables wrt referene variable
"""

import os
WDPATH = "C:\\Users\\eandr\\Documents\\1_PROJECTS\\A_PAPERS\\ST_LATAM"
os.chdir(WDPATH)

from utils import dataset_ST as ds
from utils import graphs_ST as gs
from utils import ur_tests_ST as urts
from utils import lagcorr_tests_ST as lcrts
from utils import gc_tests_ST as gcts


# Define environment
# =============================================================================
# params for tests
signif = [0.05]
p = 24  # max lag length
lag = 12  # usual lag length (1 year)
num = 3  # total variables
# label structure
prefix = "indices_"
suffix = "_data_ST"
sheet = "Import_loc" # "Import_esp" for analysis with spanish press 
# variables and countries (order alphabetically)
varname = ["sou", "pot", "epu"]
ref = "sou"  # "pot" "epu"; reference variable for GC test
country = ["AG", "BR", "CL", "CO", "MX", "PE", "LA"]
country_long = ["argentina", "brazil", "chile", "colombia", "mexico", "peru", "latam_wav"] # "latam_pca" for PCA aggregation method
country_Long = ["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru", "Latam"]
# colour schemes
colours = ["k", "g", "r", "y", "b", "m", "c"]
palettes = ["RdGy", "YlGn",  "OrRd", "YlOrBr", "PuBu", "PiYG", "BrBG"]


# Data analysis
# =============================================================================
# paths
DATA = ds.read_data(WDPATH, prefix, suffix, sheet,
                    country, country_long,)[0]
OUTPUTS = ds.read_data(WDPATH, prefix, suffix, sheet,
                       country, country_long,)[1]
# data import
data = ds.read_data(WDPATH, prefix, suffix, sheet,
                    country, country_long,)[2]
# descriptive stats and plots
stats = gs.plot_data(OUTPUTS, colours, palettes, data, country, country_Long,
                     varname, ref)

# Econometric analysis
# =============================================================================
# Joint tests: lag order and GC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

model = {}
results = {}
ic = "bic" # "aic", "hqic"
test = "f" 
sig = 0.1
# https://www.statsmodels.org/stable/vector_ar.html
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VARResults.test_causality.html#statsmodels.tsa.vector_ar.var_model.VARResults.test_causality

# potential log-diff + drop-nan for stationary time-series
# data = np.log(mdata).diff().dropna() 

# own country
for i in country:
    model["var_"+i] = VAR(data["data_"+i])
    results["var_res_"+i] = model["var_"+i].fit(maxlags=p, ic=ic) # maxlags=p, ic=ic
    #print(results["var_res_"+i].summary())
    for j in varname:
        if j == "sou":
            results["gc_res_"+i] = results["var_res_"+i].test_causality(j+i, ["pot"+i], kind=test, signif=sig)
            print("Results for "+i+" with caused variable "+j)
            print(results["gc_res_"+i].summary())
            results["gc_res_"+i] = results["var_res_"+i].test_causality(j+i, ["epu"+i], kind=test, signif=sig)
            print("Results for "+i+" with caused variable "+j)
            print(results["gc_res_"+i].summary())
            results["gc_res_"+i] = results["var_res_"+i].test_causality(j+i, ["pot"+i, "epu"+i], kind=test, signif=sig)
            print("Results for "+i+" with causing variables pot and epu"+" and caused variable "+j)
            print(results["gc_res_"+i].summary())
        if j == "pot":
            results["gc_res_"+i] = results["var_res_"+i].test_causality(j+i, ["sou"+i], kind=test, signif=sig)
            print("Results for "+i+" with caused variable "+j)
            print(results["gc_res_"+i].summary())
            results["gc_res_"+i] = results["var_res_"+i].test_causality(j+i, ["epu"+i], kind=test, signif=sig)
            print("Results for "+i+" with caused variable "+j)
            print(results["gc_res_"+i].summary())
            results["gc_res_"+i] = results["var_res_"+i].test_causality(j+i, ["sou"+i, "epu"+i], kind=test, signif=sig)
            print("Results for "+i+" with causing variables sou and epu"+" and caused variable "+j)
            print(results["gc_res_"+i].summary())
        if j == "epu":
            results["gc_res_"+i] = results["var_res_"+i].test_causality(j+i, ["sou"+i], kind=test, signif=sig)
            print("Results for "+i+" with caused variable "+j)
            print(results["gc_res_"+i].summary())
            results["gc_res_"+i] = results["var_res_"+i].test_causality(j+i, ["pot"+i], kind=test, signif=sig)
            print("Results for "+i+" with caused variable "+j)
            print(results["gc_res_"+i].summary())
            results["gc_res_"+i] = results["var_res_"+i].test_causality(j+i, ["sou"+i, "pot"+i], kind=test, signif=sig)
            print("Results for "+i+" with causing variables sou and pot"+" and caused variable "+j)
            print(results["gc_res_"+i].summary())
            
            
print(results["var_res_AG"].summary())
print(results["var_res_BR"].summary())
print(results["var_res_CL"].summary())
print(results["var_res_CO"].summary())
print(results["var_res_MX"].summary())
print(results["var_res_PE"].summary())
print(results["var_res_LA"].summary())
            
# cross country
os.chdir(WDPATH)
xls = pd.ExcelFile("Indices_tensions.xlsx")

data_all = {}
model_all = {}
results_all = {}

for j in varname:
    data_all["data_"+j] = pd.read_excel(xls, "Import_"+j, index_col=1, header=1)
    data_all["data_"+j].dropna(axis=1, how="all", inplace=True)
    model_all["var_"+j] = VAR(data_all["data_"+j])
    results_all["var_res_"+j] = model_all["var_"+j].fit(maxlags=p, ic=ic) #maxlags=p, ic=ic
    
print(results_all["var_res_sou"].summary())
print(results_all["var_res_pot"].summary())
print(results_all["var_res_epu"].summary())

ref_country = "PE"
cross_country = ["AG", "BR", "CL", "CO", "MX", "PE"]
for i in cross_country:
    for j in varname: 
        if j == "sou":
            results_all["gc_res_"+ref_country] = results_all["var_res_"+j].test_causality(j+ref_country, j+i, kind=test, signif=sig)
            print("Results for "+ref_country+ " with causing country "+i+" and caused variable "+j)
            print(results_all["gc_res_"+ref_country].summary())
        if j == "pot":
            results_all["gc_res_"+ref_country] = results_all["var_res_"+j].test_causality(j+ref_country, j+i, kind=test, signif=sig)
            print("Results for "+ref_country+ " with causing country "+i+" and caused variable "+j)
            print(results_all["gc_res_"+ref_country].summary())
        if j == "epu":
            results_all["gc_res_"+ref_country] = results_all["var_res_"+j].test_causality(j+ref_country, j+i, kind=test, signif=sig)
            print("Results for "+ref_country+ " with causing country "+i+" and caused variable "+j)
            print(results_all["gc_res_"+ref_country].summary())            

# additional results
# print(results["var_res_AG"].summary())
# print(results["var_res_AG"].plot())
# print(results["var_res_AG"].plot_acorr())
# plt.rc('figure', figsize=(7, 5))
# plt.text(0.01, 0.05, str(results["gc_res_AG"].summary()), {'fontsize': 12}, 
#          fontproperties = 'monospace')
# plt.axis('off')
# plt.tight_layout()
# plt.savefig(os.path.join(WDPATH, "gc_res_AG.png"),
#                         dpi=200, bbox_inches="tight")


# Individual tests: UR, lag order, DW, and GC
aic_res = lcrts.lag_corr_test_data(data, country, p, lag)[0]
bic_res = lcrts.lag_corr_test_data(data, country, p, lag)[1]
aic_all_res = lcrts.lag_corr_test_data(data, country, p, lag)[2]
bic_all_res = lcrts.lag_corr_test_data(data, country, p, lag)[3]
dw_res = lcrts.lag_corr_test_data(data, country, p, lag)[4]
adf_res = urts.ur_test_data(data, country, signif)[0]
kpss_res = urts.ur_test_data(data, country, signif)[1]
granger_res = gcts.gc_test_data(data, country, varname, ref,
                                signif, lag, num)[0]
granger_vec_res = gcts.gc_test_data(data, country, varname, ref,
                                    signif, lag, num)[1]
granger_mtx_res = gcts.gc_mtx_test_data(data, country, lag)


# Additional regression analysis
# =============================================================================
# import statsmodels.api as sm
# import numpy as np
# import matplotlib.pyplot as plt

# for l in range(len(varname)):
#     ref_var = str(varname[l])
#     for k in range(len(country)):
#         ref_country = str(country[k])
#         for i in range(len(country)):
#             str_country = str(country[i])
#             Y = data["data_"+ref_country][ref_var+ref_country]
#             X = data["data_"+str_country][ref_var+str_country]
#             X = sm.add_constant(X)
#             model = sm.OLS(Y,X)
#             ols = model.fit(cov_type="HC3")
#             print(ols.summary())
#             plt.rc('figure', figsize=(7, 5))
#             plt.text(0.01, 0.05, str(ols.summary()), {'fontsize': 12},
#                      fontproperties = 'monospace')
#             plt.axis('off')
#             plt.tight_layout()
#             plt.savefig(os.path.join(WDPATH, "ols_reg_"+ref_var+ref_country+"_"+ref_var+str_country+".png"),
#                         dpi=200, bbox_inches="tight")
#             plt.clf()
