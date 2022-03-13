import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests as gct

test = "ssr_ftest"
gc_matrix = {}
gc_test = {}
pred_caus_gc = {}
str_gc2 = []
gc_t = []


def gc_mtx_test_data(data, country_name, l, test=test):
    for i in range(len(country_name)):
        # shortcuts
        str_country = str(country_name[i])
        data_country = data["data_"+str_country]
        gc_matrix[str_country] = {}
        variables = data_country.columns
        # build GC matrix with p-values of test results
        # columns X are predictors and rows Y are responses
        # null of not GC, hence if reject, evidence of X GC Y
        gc_matrix[str_country] = \
            pd.DataFrame(np.zeros((len(variables), len(variables))),
                         columns=variables, index=variables)
        # save elements of df as objects to fill them with lists of p-values
        gc_matrix[str_country] = gc_matrix[str_country].astype('object')
        for c in gc_matrix[str_country].columns:
            for r in gc_matrix[str_country].index:
                test_res = gct(data_country[[r, c]],
                               maxlag=l, verbose=False)
                # round p-vals
                p_values = []
                for j in range(l):
                    p_values.append(round(test_res[j+1][0][test][1], 4))
                # if want lag with min p-val: min_p_value = np.min(p_values)
                # keep only min p-vals in matrix
                gc_matrix[str_country].loc[r, c] = p_values  # min_p_value
        gc_matrix[str_country].columns = [var + '_X' for var in variables]
        gc_matrix[str_country].index = [var + '_Y' for var in variables]

    return gc_matrix


def gc_test_data(data, country_name, varname, ref, signif, l, num, test=test):
    for i in range(len(country_name)):
        # shortcuts
        str_country = str(country_name[i])
        data_country = data["data_"+str_country]
        str_gc = "gc_"+str_country
        gc_test[str_gc] = {}
        # GC tests - last variable (ref) not GC first variable (rest)
        for n in range(len(varname)):
            # shortcuts
            str_gc2.append(varname[n]+"_"+str_country+"_"+ref
                           + "_"+str_country)
            gc_test[str_gc][str_gc2[(i*num)+(n)]] = {}
            test_res = gct(data_country[[varname[n]+str_country,
                                         ref+str_country]],
                           maxlag=l,
                           verbose=False)
            for j in range(l):
                # shortcuts
                str_lag = "lag_"+str(j+1)
                str_gc3 = str_gc+"_"+str_gc2[(i*num)+(n)]+"_"+str_lag
                gc_t.append(round(test_res[j+1][0][test][1], 4))
                gc_test[str_gc][str_gc2[(i*num)+(n)]][str_lag] = \
                    gc_t[(i*num*l)+(n*l)+(j)]
                # keep only GC results
                for k in range(len(signif)):
                    pred_caus_gc[str_gc3] = \
                        gc_test[str_gc][str_gc2[(i*num)+(n)]][str_lag] \
                        <= signif[k]
                    if not pred_caus_gc[str_gc3]:
                        del pred_caus_gc[str_gc3]

    return pred_caus_gc, gc_test
