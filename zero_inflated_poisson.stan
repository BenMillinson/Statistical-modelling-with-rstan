
data {
  int<lower=0> N;               // Number of observations
  int<lower=0> K;               // Number of predictors
  matrix[N, K] X;               // Predictor matrix
  int<lower=0> y[N];            // Count outcome variable
}
parameters {
  vector[K] beta;               // Coefficients for predictors
  real alpha;                   // Intercept for Poisson regression
  real<lower=0, upper=1> pi;   // Probability of zero inflation
}
model {
  // Priors
  beta ~ normal(0, 1);
  alpha ~ normal(0, 1);
  pi ~ beta(1, 1);             // Uniform prior for zero inflation parameter

  // Likelihood
  for (n in 1:N) {
    if (y[n] == 0) {
      y[n] ~ bernoulli(pi);   // Zero-inflation model
    } else {
      y[n] ~ poisson_log(alpha + X[n] * beta);  // Poisson model for counts
    }
  }
}

