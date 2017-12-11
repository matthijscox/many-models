library(data.table)
library(microbenchmark)

# Load data
dat <- fread('https://raw.githubusercontent.com/matthijscox/many-models/master/many_models_data.csv')

## Function definitions
create.groups <- function( inputData, by ) {

  # Create a factor array of the grouping columns so we only need to loop across this vector
  ncols <- length(by)
  for( ci in ncols:1) {
    factors <- factor(inputData[, by[ci]], exclude = NULL) # Create factor array, also include NA as levels
    if (ci == ncols) {
      groupedIDs <- as.integer(factors)
    }
    else { # Create a unique positive integer.
      groupedIDs <- groupedIDs * nlevels(factors) + as.integer(factors)
    }
  }
  return( match(groupedIDs, unique.default(groupedIDs)) )
}

create.matrix <- function(nrow,ncol) {
  x <- matrix()
  length(x) <- nrow*ncol
  dim(x) <- c(nrow,ncol)
  x
}

create.model.matrix <- function(A,B,poly_order=3) {
  
  npoly <- (poly_order+1)*(poly_order+2)/2
  
  # pre-allocate matrix
  M <- create.matrix(dim(A)[1],npoly)   
  
  # Fill the model matrix
  M[,1] <- A*0+1
  for( pow in 1:poly_order) {
    for (pow.y in 0:pow) {
      pow.x <- pow - pow.y
      M[,pow*(pow+1)/2+pow.y+1] <- A^pow.x*B^pow.y
    }
  }
  M

}

calc_poly3_residuals <- function(A,B,Z) {
  
  # construct model matrix
  M <- cbind(A*0+1, A, B, A*B, A^2, A^2*B, B^2, A*B^2, A^3, B^3)
  
  # calculate residuals 
  resid <- Z - M %*% solve(qr(M, LAPACK = TRUE), Z)
}


calc_poly3_residuals_lm <- function(A,B,Z) {
  
  # construct model matrix
  M <- cbind(A*0+1, A, B, A*B, A^2, A^2*B, B^2, A*B^2, A^3, B^3)
  
  # calculate residuals 
  mod <- .lm.fit( M, Z )
  resid <- Z - M %*% mod$coefficients
}

calc_poly3_residuals_lm_no_cbind <- function(A,B,Z) {
  
  # construct model matrix
  M <- create.model.matrix(as.matrix(A),as.matrix(B),3)
  
  # calculate residuals 
  mod <- .lm.fit( M, Z )
  resid <- Z - M %*% mod$coefficients
}

calc_residuals_per_group <- function(dat,groups,fun) {
  groupedIDs <- create.groups(dat,by=groups)
  nGroups <- length( unique(groupedIDs) )                   # Number of unique IDs
  idx <- split(seq_along(groupedIDs), groupedIDs)           # Find indices of the various unique groupedIDs
  Amat <- as.matrix(dat$A)
  Bmat <- as.matrix(dat$B)
  Zmat  <- as.matrix(dat$Z)
  res <- vector(mode='numeric', length=nrow(dat))
  
  for ( gi in 1:nGroups ) { 
    # get index locations of the group?
    idxLoc <- idx[[gi]]
    # no dropping == speed
    A <- Amat[idxLoc,,drop=F]
    B <- Bmat[idxLoc,,drop=F]
    Z <- Zmat[idxLoc,,drop=F]
    
    # do the fit and assign back
    res[idxLoc] <- fun(A,B,Z)
    
  }
  
  return( res )
}

calc_residuals_by_datatable <- function(dat,fun) {
  dat[,resid:=fun(A,B,Z), by=.(num_group,cat_group)]
}
  
calc_residuals_with_by <- function(x,groups) { 
  
  groupedIDs <- create.groups(x,by=groups)
  
  df_fun <- function(x) {
    calc_poly3_residuals_lm(x$A,x$B,x$Z)
  }
  res <- by(dat,groupedIDs, df_fun)
  res <- unlist(res)
} 

## Doing the timing
df <- as.data.frame(dat)

forloop.timing <- microbenchmark(calc_residuals_per_group(df,c('num_group','cat_group'),calc_poly3_residuals_lm),times=100)
datatable.qr.timing <- microbenchmark(calc_residuals_by_datatable(dat,calc_poly3_residuals),times=100)
datatable.lm.timing <- microbenchmark(calc_residuals_by_datatable(dat,calc_poly3_residuals_lm),times=100)
by.lm.timing <- microbenchmark(calc_residuals_with_by(df,c('num_group','cat_group')),times=10)

# just a verification
res <- calc_residuals_per_group(df,c('num_group','cat_group'),calc_poly3_residuals_lm_no_cbind)
sqrt(sum(res^2)/length(res))-0.878223092545709

## plotting
library(tidyverse)
t <- tibble(method = c('data.table + qr.solve','for-loop + lm.fit','data.table + lm.fit','by + lm.fit'), 
       timing = c(mean(datatable.qr.timing$time)/1e6,mean(forloop.timing$time)/1e6,mean(datatable.lm.timing$time)/1e6,mean(by.lm.timing$time)/1e6))

# reorder method levels by timing
t$method <- factor(t$method,t$method[order(t$timing, decreasing = F)])

(g2 <- ggplot(t,aes(method,timing)) + 
  geom_bar(stat = 'identity', fill = '#36ba97', width=0.85) +
  geom_text(aes(label=sprintf("%0.0f ms", timing)), hjust=-.1, size=4, colour='#36ba97') +
  labs(title = 'Time per Method', x='', y='') +
  coord_flip() + 
  theme_minimal() +
  ylim(0,5000) + 
  theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(), 
          axis.text.x = element_blank(), 
          axis.text.y = element_text(size=11), 
          legend.position="none"))

file_path = "C:/Localdata/Repositories/many-models/trunk/"
ggsave('many_models_forloop.jpg', plot = g2, device = NULL, path = file_path,
       scale = 1, width = 160, height = 80, units = 'mm',
       dpi = 600, limitsize = TRUE) 
