
data {
  int<lower=0> N;               // Number of observations
  int<lower=0> K;               // Number of predictors
  matrix[N, K] X;               // Predictor matrix
  int<lower=0> y[N];            // Count outcome variable
}
parameters {
  vector[K] beta;               // Coefficients for predictors
  real alpha;                   // Intercept
}
model {
  // Priors
  beta ~ normal(0, 1);
  alpha ~ normal(0, 1);
  
  // Likelihood
  y ~ poisson_log(alpha + X * beta);
}

