
data {
  int<lower=0> N1;              // Number of observations in group 1
  int<lower=0> N2;              // Number of observations in group 2
  vector[N1] x1;                // Observations from group 1
  vector[N2] x2;                // Observations from group 2
}
parameters {
  real mu1;                     // Mean of group 1
  real mu2;                     // Mean of group 2
  real<lower=0> sigma1;         // Standard deviation of group 1
  real<lower=0> sigma2;         // Standard deviation of group 2
}
model {
  // Priors
  mu1 ~ normal(0, 10);
  mu2 ~ normal(0, 10);
  sigma1 ~ normal(0, 5);
  sigma2 ~ normal(0, 5);
  
  // Likelihood
  x1 ~ normal(mu1, sigma1);
  x2 ~ normal(mu2, sigma2);
}
generated quantities {
  real mu_diff = mu1 - mu2;     // Difference in means
}

