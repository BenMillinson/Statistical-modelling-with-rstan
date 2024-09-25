
data {
  int<lower=0> J;         // number of groups
  int<lower=0> N;         // number of data points
  int<lower=1, upper=J> group[N];  // group indicator
  vector[N] x;            // predictor
  vector[N] y;            // outcome
}
parameters {
  vector[J] alpha;          // group intercepts
  real beta;              // common slope
  real mu_alpha;          // prior mean for intercepts
  real<lower=0> sigma;    // standard deviation of observations
  real<lower=0> sigma_alpha;  // standard deviation of intercepts
}
model {
  mu_alpha ~ normal(0, 10);
  alpha ~ normal(mu_alpha, sigma_alpha);
  beta ~ normal(0, 10);
  y ~ normal(alpha[group] + beta * x, sigma);
}

