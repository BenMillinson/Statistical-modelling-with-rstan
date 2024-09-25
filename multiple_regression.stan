
data {
  int<lower=0> N;            // Number of data points
  int<lower=0> K;            // Number of predictors (including intercept)
  matrix[N, K] X;            // Predictor matrix
  vector[N] y;               // Outcome vector
}
parameters {
  vector[K] beta;            // Coefficients for predictors (including intercept)
  real<lower=0> sigma;       // Standard deviation of the residuals
}
model {
  y ~ normal(X * beta, sigma);  // Likelihood
}

