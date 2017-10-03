using DataFrames
using GLM
using Requests

resp = get("https://raw.githubusercontent.com/matthijscox/many-models/master/many_models_data.csv")
df = readtable(IOBuffer(resp.data))

# Using the GLM package
function calc_poly_residuals(df)
    # Q: how to input quadratic terms, such as A^2 and A^2*B?
    OLS = fit(LinearModel,@formula(Z ~ A*B), df)
    resid = df[:Z]-predict(OLS)
end

# 2D polynomial model matrix
function model_matrix_poly(x,y,poly_order=3)
    ndat = length(x)
    npoly = (poly_order+1)*(poly_order+2)
    M = ones(ndat,npoly/2)
    for pow = 1:poly_order
        for pow_y = 0:pow
            pow_x = pow-pow_y
            M[:,round(Int,pow*(pow+1)/2+pow_y+1)] = (x.^pow_x .* y.^pow_y)
        end
    end
    return M
end

# using model matrices directly
function calc_poly_residuals_with_matrices(df)        
    # Using a model matrix and mldivide
    M = model_matrix_poly(df[:A],df[:B])
    y = convert(Array,df[:Z])
    c = M \ y
    resid = y-M*c
end

# quite impressive timing for dataframes (0.7 seconds using matrix approach, 1.7s using GLM)

@time resid = by(df,[:num_group,:cat_group],df -> DataFrame(resid = calc_poly_residuals(df)))[:resid]
@time resid = by(df,[:num_group,:cat_group],df -> DataFrame(resid = calc_poly_residuals_with_matrices(df)))[:resid]

# root-mean-square compares well to Matlab result
mat_diff = sqrt(sum(resid.^2)/size(resid,1))-0.878223092545709
print("Julia difference to reference: $mat_diff")

# reference:
# https://juliadata.github.io/DataFrames.jl/stable/man/split_apply_combine/