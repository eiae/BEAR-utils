from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson as dw

aic_lag = {}
bic_lag = {}
opt_aic = {}
opt_bic = {}
VARs = {}
dw_test = {}


def lag_corr_test_data(data, country_name, p, l):
    for i in range(len(country_name)):
        # shortcuts
        str_country = str(country_name[i])
        data_country = data["data_"+str_country]
        VAR_country = VAR(data_country)
        aic_lag[str_country] = {}
        bic_lag[str_country] = {}
        opt_aic[str_country] = {}
        opt_bic[str_country] = {}
        res_aic = []
        res_bic = []
        # Lag length order information criteria - AIC / BIC
        # AIC better when false negative more misleading than false positive
        # BIC better when false positive more misleading than false negative
        for j in range(1, p+1):
            # shortcuts
            str_aic = "aic_lag_"+str(j)
            str_bic = "bic_lag_"+str(j)
            try:
                res_VAR = VAR_country.fit(j)
                aic_lag[str_country][str_aic] = round(res_VAR.aic, 4)
                bic_lag[str_country][str_bic] = round(res_VAR.bic, 4)
                opt_aic[str_country][str_aic] = round(res_VAR.aic, 4)
                opt_bic[str_country][str_bic] = round(res_VAR.bic, 4)
                res_aic.append(round(res_VAR.aic, 4))
                res_bic.append(round(res_VAR.bic, 4))
            except:
                continue
        # find value closest to zero (optimality criterion)
        min_aic = min(res_aic, key=abs)
        min_bic = min(res_bic, key=abs)
        keys_aic = list(opt_aic[str_country].keys())
        keys_bic = list(opt_bic[str_country].keys())
        # keep only lag order associated to min value
        for k in range(len(keys_aic)):
            if not opt_aic[str_country][keys_aic[k]] == min_aic:
                del opt_aic[str_country][keys_aic[k]]
        for k in range(len(keys_bic)):
            if not opt_bic[str_country][keys_bic[k]] == min_bic:
                del opt_bic[str_country][keys_bic[k]]
        # DW test for serial correlation of residuals
        # closer to 2, no significant serial correlation
        # closer to 0, positive serial correlation
        # closer to 4, negative serial correlation
        # evaluate test on AIC optimal lag lenght
        # if serially correlated, add more lags
        VARs[str_country] = \
            VAR_country.fit(int(list(aic_lag[str_country].keys())[l-1][-1]))
        dw_test[str_country] = {}
        res_dw = dw(VARs[str_country].resid)
        for c, d in zip(data_country.columns, res_dw):
            dw_test[str_country][c] = d

    return opt_aic, opt_bic, aic_lag, bic_lag, dw_test
