
data {
  int<lower=0> N;               // Number of trials
  int<lower=0> hits;            // Number of hits
  int<lower=0> misses;          // Number of misses
  int<lower=0> false_alarms;    // Number of false alarms
  int<lower=0> correct_rejections; // Number of correct rejections
}
parameters {
  real<lower=0> d_prime;        // Sensitivity parameter
  real c;                       // Criterion parameter
}
model {
  real p_hit;                   // Probability of a hit
  real p_fa;                    // Probability of a false alarm
  
  // Priors
  d_prime ~ normal(0, 1);
  c ~ normal(0, 1);

  // Calculate probabilities based on d' and c
  p_hit = 1 - normcdf(c + d_prime);  // p(hit) = 1 - CDF of normal distribution
  p_fa = normcdf(c);                 // p(false alarm) = CDF of normal distribution

  // Likelihood
  hits ~ binomial(N, p_hit);
  false_alarms ~ binomial(N, p_fa);
  misses ~ binomial(N, 1 - p_hit);
  correct_rejections ~ binomial(N, 1 - p_fa);
}

