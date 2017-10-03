library(data.table)

###############################################################################
# data.table + lm
# 6 seconds

dat <- fread('https://raw.githubusercontent.com/matthijscox/many-models/master/many_models_data.csv')

# HOWA3 model definition, quote was needed for environment issues with lm in the datatable
model_func <- quote(lm(Z ~ poly(A, B, degree=3, raw=TRUE)))

start.time <- Sys.time()

dat[,resid:=residuals(eval(model_func)), by=.(num_group,cat_group)]

end.time <- Sys.time()
(time.taken <- end.time - start.time)

# RMS comparison
sqrt(sum(dat$resid^2)/nrow(dat))-0.878223092545709

###############################################################################
# data.table + direct matrix solver
# 0.5 seconds

calc_poly3_residuals <- function(A,B,Z) {

  # construct model matrix
  M <- cbind(A*0+1, A, B, A*B, A^2, A^2*B, B^2, A*B^2, A^3, B^3)
  
  # calculate residuals 
  resid <- Z - M %*% solve(qr(M, LAPACK = TRUE), Z)
}

start.time <- Sys.time()
dat[,resid:=calc_poly3_residuals(A,B,Z), by=.(num_group,cat_group)]
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

# RMS comparison
sqrt(sum(dat$resid^2)/nrow(dat))-0.878223092545709

###############################################################################
# R for Data Science, the chapter on many models describes the following way of working
# 12 seconds

library('tidyverse')
library('modelr')

t <- as_tibble(dat)

model_func <- function(x) { lm(Z ~ poly(A, B, degree=3, raw=TRUE), data = x) }

start.time <- Sys.time()
t2 <- t %>% group_by(num_group,cat_group) %>% nest() %>% 
  mutate(model = map(data,model_func) ) %>%  
  mutate(residuals = map2(data, model, add_residuals) ) %>% 
  unnest(residuals)
end.time <- Sys.time()
(time.taken <- end.time - start.time)

sqrt(sum(t2$resid^2)/nrow(t2))-0.878223092545709

###############################################################################
# R base functions + lm
# 9 seconds...

dat <- as.data.frame(dat)

# have to get the group indices for unique direction/wafer groups
# using dplyr for now, don't know a nice base function
grpidx = dat %>% group_by(num_group,cat_group) %>% group_indices

start.time <- Sys.time()

# Define the model function
model_func <- function(x) { lm(Z ~ poly(A, B, degree=3, raw=TRUE), data = x) }
  
# Model by group indices
tmpx <- by(dat,grpidx, model_func)
  
# Extract the coefficients and residuals per group
res <- sapply(tmpx, residuals)
res <- unlist(res)  

end.time <- Sys.time()
(time.taken <- end.time - start.time)

sqrt(sum(res^2)/length(res))-0.878223092545709

###############################################################################
# R base functions + matrix solver
# 3.4 seconds...

start.time <- Sys.time()

df_calc_poly3_residuals <- function(x) { 
  calc_poly3_residuals(x$A,x$B,x$Z)
}

# Directly calculate residuals
res <- by(dat,grpidx, df_calc_poly3_residuals)
res <- unlist(res)

end.time <- Sys.time()
(time.taken <- end.time - start.time)

sqrt(sum(res^2)/length(res))-0.878223092545709
