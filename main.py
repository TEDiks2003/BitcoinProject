import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier

# import train data
df = pd.read_csv("BitcoinTraining.csv")


# func to flip sign
def flip_tar(pred, ispredinc ,ispreddec, isinc, isdec):
    if (isinc and ispreddec) or (isdec and ispredinc):
        return pred*-1
    else:
        return pred


# func to flip signs for whole list
def flip_tar_list(pred_list, inc_list, dec_list):
    pred_array = np.array(pred_list)
    pred_inc_list = np.where(pred_array > 0, True, False)
    pred_dec_list = np.where(pred_array < 0, True, False)
    flippedlist = [flip_tar(p, pI, pD, i, d) for p, pI, pD, i, d in zip(pred_list, pred_inc_list, pred_dec_list, inc_list, dec_list)]
    return flippedlist


# finding imbalance avg
bid_vol_row_names = [f"Bid{x}#" for x in range(1, 11)]
ask_vol_row_names = [f"Ask{x}#" for x in range(1, 11)]
df['volume_bid'] = df.loc[:, bid_vol_row_names].sum(axis=1)
df['volume_ask'] = df.loc[:, ask_vol_row_names].sum(axis=1)
df['imbalance_avg'] = (df['volume_bid'] - df['volume_ask']) / (df['volume_bid'] + df['volume_ask'])

data = df.drop(["volume_bid", "volume_ask"], axis=1)

predict = "TARGET"
X = np.array(data.drop([predict], axis=1))
Y = np.array(data[predict])

# finds if target is increasing or decreasing
y_isIncreasing = np.where(Y > 0, True, False)
y_isDecreasing = np.where(Y < 0, True, False)

# gets test data
testDF = pd.read_csv("D:\Coding Files\Bitcoin\BitcoinTest.csv")
testDF['volume_bid'] = testDF.loc[:, bid_vol_row_names].sum(axis=1)
testDF['volume_ask'] = testDF.loc[:, ask_vol_row_names].sum(axis=1)
testDF['imbalance_avg'] = (testDF['volume_bid'] - testDF['volume_ask']) / (testDF['volume_bid'] + testDF['volume_ask'])

x_test = np.array(testDF.drop(["volume_bid", "volume_ask"], axis=1))

# RFC to predict whether it will be increasing or decreasing
forestInc = RandomForestClassifier()
forestInc.fit(X, y_isIncreasing)

forestDec = RandomForestClassifier()
forestDec.fit(X, y_isDecreasing)

y_pred_inc = forestInc.predict(x_test)
y_pred_dec = forestDec.predict(x_test)

# Lasso Regressor
l_reg = Lasso()
l_reg.fit(X, Y)
pred_lasso = l_reg.predict(x_test)

# Bayesian Ridge Regressor
br = BayesianRidge()
br.fit(X, Y)
pred_bay = br.predict(x_test)

# Linear Regressor
linear = LinearRegression()
linear.fit(X, Y)
pred_lin = linear.predict(x_test)

# Ridge Regressors
r_reg = Ridge(solver = 'svd', random_state = 10)
r_reg1 = Ridge(solver = 'lsqr', random_state = 10)
r_reg.fit(X, Y)
r_reg1.fit(X, Y)
pred_ridge_svd = r_reg.predict(x_test)
pred_ridge_lsqr = r_reg1.predict(x_test)

# Flip depending on predicted increasing or decreasing
pred_lasso_f = flip_tar_list(pred_lasso, y_pred_inc, y_pred_dec)
pred_bay_f = flip_tar_list(pred_bay, y_pred_inc, y_pred_dec)
pred_lin_f = flip_tar_list(pred_lin, y_pred_inc, y_pred_dec)
pred_ridge_svd_f = flip_tar_list(pred_ridge_svd, y_pred_inc, y_pred_dec)
pred_ridge_lsqr_f = flip_tar_list(pred_ridge_lsqr, y_pred_inc, y_pred_dec)

# Gets mean prediction
pred_mean = [(a+b+c+d+e)/5 for a, b, c, d, e in zip(pred_lasso_f, pred_bay_f, pred_lin_f, pred_ridge_svd_f, pred_ridge_lsqr_f)]

testDF["TARGET"] = pred_mean

# Export
testDF.to_csv(r'BitcoinEstimates.csv', index=False, header=True)

