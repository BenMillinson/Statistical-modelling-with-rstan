
data {
  int<lower=0> N;               // Number of observations
  int<lower=0> K;               // Number of predictors
  matrix[N, K] X;               // Predictor matrix
  int<lower=0> y[N];            // Count outcome variable
}
parameters {
  vector[K] beta;               // Coefficients for predictors
  real alpha;                   // Intercept for Poisson part
  real<lower=0> phi;            // Dispersion parameter for Negative Binomial
  real<lower=0, upper=1> pi;   // Probability of zero inflation
}
model {
  vector[N] lambda;             // Rate parameter for the Negative Binomial component
  
  // Priors
  beta ~ normal(0, 1);
  alpha ~ normal(0, 1);
  phi ~ gamma(1, 1);            // Gamma prior for the dispersion parameter
  pi ~ beta(1, 1);              // Uniform prior for zero inflation parameter
  
  // Calculate lambda
  lambda = exp(alpha + X * beta);
  
  // Likelihood
  for (n in 1:N) {
    if (y[n] == 0) {
      y[n] ~ bernoulli(pi);   // Zero-inflation model
    } else {
      y[n] ~ neg_binomial_2(lambda, phi);  // Negative Binomial model for counts
    }
  }
}

