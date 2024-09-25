
data {
  int<lower=0> N;               // Number of observations
  int<lower=0> K;               // Number of predictors
  matrix[N, K] X;               // Predictor matrix
  int<lower=0> y[N];            // Count outcome variable
}
parameters {
  vector[K] beta;               // Coefficients for predictors
  real alpha;                   // Intercept
  real<lower=0> phi;            // Dispersion parameter
}
model {
  vector[N] lambda;             // Rate parameter for the Poisson component
  
  // Priors
  beta ~ normal(0, 1);
  alpha ~ normal(0, 1);
  phi ~ gamma(1, 1);            // Gamma prior for the dispersion parameter
  
  // Calculate lambda
  lambda = exp(alpha + X * beta);
  
  // Likelihood
  y ~ neg_binomial_2(lambda, phi);  // Negative Binomial distribution
}

