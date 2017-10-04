import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import timeit

# read csv from url
df = pd.read_csv("https://raw.githubusercontent.com/matthijscox/many-models/master/many_models_data.csv")

# using the statsmodels package
# defining a simple linear model for testing, TODO building a 3rd order polynomial
def calc_poly_residuals_ols(df):
    f = 'Z ~ A + B'
    resid = sm.ols(formula=f, data=df).fit().resid
    return resid

# using numpy.lstsq with handcrafted 3rd order polynomial model matrix
def calc_poly_residuals_lstsq(X, Y, Z):
    # 3rd order 2D polynomial model matrix
    M = np.array([X * 0 + 1, X, Y, X * Y, X ** 2, X ** 2 * Y, Y ** 2, X * Y ** 2, X ** 3, Y ** 3]).T
    # M = model_matrix_poly(X,Y) # slightly slower

    # do the fit
    coeff = np.linalg.lstsq(M, Z)  # using the least-square regression function
    coeff = coeff[0]

    # calculate residuals
    resid = np.dot(M, coeff) - Z
    return resid

def calc_poly_residuals_lstsq_df(df):
    X = df['A'].values
    Y = df['B'].values
    Z = df['Z'].values
    resid = calc_poly_residuals_lstsq(X, Y, Z)
    return resid

# ------------ Pandas + statsmodels approach -------------
# see: https://pandas.pydata.org/pandas-docs/stable/groupby.html
# already taking over 22 seconds...

start = timeit.default_timer()

# doing the groupby
df_grouped = df.groupby(['cat_group', 'num_group'])

# here we go, using apply
tst = df_grouped.apply(calc_poly_residuals_ols)

stop = timeit.default_timer()

print('Pandas + statsmodels timing: {:.2f} seconds'.format(stop - start))

# ------------- Pandas + numpy approach --------------
# using numpy lstlq
# 1.7 seconds

start = timeit.default_timer()
df_grouped = df.groupby(['cat_group', 'num_group']) # group by
tst = df_grouped.apply(calc_poly_residuals_lstsq_df)   # apply
stop = timeit.default_timer()

print('Pandas + lstsq timing: {:.2f} seconds'.format(stop - start))

# flatten doubly nested array
resid = np.array([vec for arr in tst.values for vec in arr])

# Root-mean-square comparison to Matlab reference (should be << 1):
print('Pandas difference with reference: {:.2e}'.format(np.sqrt(sum(resid**2)/len(resid))-0.878223092545709))

# --------------- Numpy only approach -----------------
# construct unique groups (... need numpy 1.13 for unique rows)
# result: 3.3 seconds

start = timeit.default_timer()

# constructing the groups by hand
num_size = max(df['num_group'])
grp = (pd.factorize(df['cat_group'].values)[0] + 1) * num_size + df['num_group'].values
grp_ids = np.unique(grp)

# initialize residuals
resid = np.zeros(df['Z'].values.shape)

# get the values into arrays
A = df['A'].values
B = df['B'].values
Z = df['Z'].values

for g in grp_ids:
    idx = grp == g
    # select the single group
    # putting the dataframe calls outside the for loop is much faster
    X = A[idx]
    Y = B[idx]
    Zi = Z[idx]

    # assign into residuals
    resid[idx] = calc_poly_residuals_lstsq(X, Y, Zi)

stop = timeit.default_timer()
print('Numpy timing: {:.2f} seconds'.format(stop - start))

# Root-mean-square comparison to Matlab reference:
print('Numpy difference with reference: {:.2e}'.format(np.sqrt(sum(resid ** 2) / len(resid)) - 0.878223092545709))
