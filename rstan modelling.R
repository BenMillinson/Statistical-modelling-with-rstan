
library(tidyverse)
library(magrittr)
library(rstan)
library(parallel)
detectCores()


#linear model example ------------------------------------------------------------

#define the stand model

stan_code <- "
data {
  int<lower=0> N;          // number of data points
  vector[N] x;             // predictor
  vector[N] y;             // outcome
}
parameters {
  real alpha;              // intercept
  real beta;               // slope
  real<lower=0> sigma;     // standard deviation
}
model {
  y ~ normal(alpha + beta * x, sigma);  // likelihood
}
"

writeLines(stan_code, "linear_regression.stan")

#---simulate data

# Simulate some data
set.seed(123)
N <- 100
x <- rnorm(N, 10, 2)
y <- 3 + 2 * x + rnorm(N, 0, 1)

# Prepare data for Stan
data_list <- list(N = N, x = x, y = y)

# Fit the model
fit <- stan(file = "linear_regression.stan", data = data_list, iter = 2000, chains = 4)

# Print the results
print(fit)

# Plot the results
plot(fit)


# multiple regression example ---------------------------------------------

# Load required libraries
library(rstan)
library(ggplot2)
library(tidyr)

# Simulate data
set.seed(123)

# Number of data points and predictors (excluding intercept)
N <- 100
K <- 3

# Simulate predictor matrix with intercept included
X <- cbind(1, matrix(rnorm(N * K), N, K))  # Adding a column of ones for the intercept

# True coefficients (including intercept)
true_beta <- c(3, 1, 2, -1)  # Intercept and coefficients for predictors

# Standard deviation of the residuals
sigma <- 1

# Simulate the outcome variable
y <- X %*% true_beta + rnorm(N, 0, sigma)

# Ensure y is a vector (should be by default)
y <- as.vector(y)

# Prepare data for Stan
data_list <- list(
  N = N,
  K = ncol(X),  # Number of predictors, including intercept
  X = X,
  y = y
)

# Define the Stan model code
stan_model_code <- "
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
"

# Write the Stan model code to a file
writeLines(stan_model_code, "multiple_regression.stan")

# Fit the model
fit <- stan(file = "multiple_regression.stan", data = data_list, iter = 2000, chains = 4)

# Print the results
print(fit)


# t-test ------------------------------------------------------------------

# Load required libraries
library(rstan)
library(ggplot2)
library(tidyr)

# Simulate data
set.seed(123)
N1 <- 30
N2 <- 30
group1 <- rnorm(N1, mean = 75, sd = 10)  # Group 1 data
group2 <- rnorm(N2, mean = 80, sd = 10)  # Group 2 data

# Prepare data for Stan
data_list <- list(
  N1 = N1,
  N2 = N2,
  x1 = group1,
  x2 = group2
)

# Define the Stan model code
stan_model_code <- "
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
"

# Write the Stan model code to a file
writeLines(stan_model_code, "bayesian_t_test.stan")

# Compile the Stan model
stan_model <- stan_model(file = "bayesian_t_test.stan")

# Fit the model
fit <- sampling(stan_model, data = data_list, iter = 2000, chains = 4)

# Print the results
print(fit)



# ANOVA -------------------------------------------------------------------

# Load required libraries
library(rstan)
library(ggplot2)
library(tidyr)

# Simulate data
set.seed(123)
N <- 90                      # Total number of observations
K <- 3                       # Number of groups

# Simulate group means and standard deviation
true_mu <- c(10, 15, 20)    # True group means
sigma <- 5                  # Standard deviation within groups

# Generate data
group <- rep(1:K, each = N / K)
y <- c(
  rnorm(N / K, mean = true_mu[1], sd = sigma),
  rnorm(N / K, mean = true_mu[2], sd = sigma),
  rnorm(N / K, mean = true_mu[3], sd = sigma)
)

# Prepare data for Stan
data_list <- list(
  N = N,
  K = K,
  group = group,
  y = y
)

# Define the Stan model code
stan_model_code <- "
data {
  int<lower=0> N;                // Total number of observations
  int<lower=0> K;                // Number of groups
  int<lower=1, upper=K> group[N]; // Group indicator for each observation
  vector[N] y;                   // Outcome variable
}
parameters {
  real mu[K];                    // Group means
  real<lower=0> sigma;           // Standard deviation within groups
}
model {
  // Priors
  mu ~ normal(0, 10);
  sigma ~ normal(0, 5);
  
  // Likelihood
  y ~ normal(mu[group], sigma);
}
generated quantities {
  matrix[K, K] mu_diff;
  for (i in 1:K) {
    for (j in 1:K) {
      mu_diff[i, j] = mu[i] - mu[j];
    }
  }
}
"

# Write the Stan model code to a file
writeLines(stan_model_code, "bayesian_anova.stan")

# Compile the Stan model
stan_model <- stan_model(file = "bayesian_anova.stan")

# Fit the model
fit <- sampling(stan_model, data = data_list, iter = 2000, chains = 4)

# Print the results
print(fit)


#hierarchical model example -----------------------------------------

stan_code <- "
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
"
writeLines(stan_code, "hierarchical_model.stan")


# Simulate some data
set.seed(123)
J <- 10
N <- 100
group <- sample(1:J, N, replace = TRUE)
x <- rnorm(N, 10, 2)
alpha <- rnorm(J, 0, 2)
beta <- 3
sigma <- 1
y <- alpha[group] + beta * x + rnorm(N, 0, sigma)

# Prepare data for Stan
data_list <- list(J = J, N = N, group = group, x = x, y = y)

# Fit the model
fit <- stan(file = "hierarchical_model.stan", data = data_list, iter = 2000, chains = 4)

# Print the results
print(fit)

# Plot the results
plot(fit)

# Extract posterior samples
posterior_samples <- extract(fit)

# Inspect the structure of posterior samples
str(posterior_samples)

library(ggplot2)

# Example: Plot posterior distribution of alpha
alpha_samples <- posterior_samples$alpha
alpha_df <- as.data.frame(alpha_samples)

# Convert to long format for ggplot2
alpha_long <- gather(alpha_df, key = "group", value = "alpha")

# Plot posterior distribution of alpha
ggplot(alpha_long, aes(x = alpha, fill = group)) +
  geom_density(alpha = 0.5) +
  labs(title = "Posterior Distribution of Group Intercepts", x = "alpha", y = "Density")



# binary logistic regression ----------------------------------------------

# Load required libraries
library(rstan)
library(ggplot2)
library(tidyr)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Simulate data
N <- 100  # Number of observations
K <- 2    # Number of predictors

# Simulate predictors (including intercept)
X <- cbind(1, matrix(rnorm(N * (K - 1)), ncol = K - 1))  # Add intercept as the first column

# True coefficients
true_beta <- c(1, -2)  # Coefficients for predictors
true_alpha <- 0.5     # Intercept

# Simulate binary outcome
logit_p <- X %*% true_beta + true_alpha
p <- 1 / (1 + exp(-logit_p))
y <- rbinom(N, 1, p)

# Prepare data for Stan
data_list <- list(
  N = N,
  K = K,
  X = X,
  y = y
)

# Define Stan model code
stan_model_code <- "
data {
  int<lower=0> N;               // Number of observations
  int<lower=0> K;               // Number of predictors (excluding intercept)
  matrix[N, K] X;               // Predictor matrix
  int<lower=0, upper=1> y[N];  // Binary outcome variable
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
  y ~ bernoulli_logit(alpha + X * beta);
}
"

# Write the Stan model code to a file
writeLines(stan_model_code, "logistic_regression.stan")

# Compile the Stan model
stan_model <- stan_model(file = "logistic_regression.stan")

# Fit the model
fit <- sampling(stan_model, data = data_list, iter = 2000, chains = 4, cores = 2)

# Print the results
print(fit)

# Extract posterior samples
posterior_samples <- extract(fit)

# Convert to data frames
beta_df <- as.data.frame(posterior_samples$beta)
alpha_df <- as.data.frame(posterior_samples$alpha)

# Reshape for plotting
beta_long <- pivot_longer(beta_df, cols = everything(), names_to = "predictor", values_to = "value")
alpha_long <- alpha_df %>% rename(value = V1) %>% mutate(predictor = "Intercept")

# Plot posterior distributions
ggplot(beta_long, aes(x = value, fill = predictor)) +
  geom_density(alpha = 0.5) +
  labs(title = "Posterior Distributions of Coefficients", x = "Coefficient Value", y = "Density") +
  theme_minimal()

# Plot posterior distribution of alpha
ggplot(alpha_long, aes(x = value)) +
  geom_density(fill = "lightblue", alpha = 0.5) +
  labs(title = "Posterior Distribution of Intercept", x = "Intercept", y = "Density") +
  theme_minimal()



# poisson regression ------------------------------------------------------

# Load required libraries
library(rstan)
library(ggplot2)
library(tidyr)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Simulate data
N <- 100  # Number of observations
K <- 2    # Number of predictors

# Simulate predictors (including intercept)
X <- cbind(1, matrix(rnorm(N * (K - 1)), ncol = K - 1))  # Add intercept as the first column

# True coefficients
true_beta <- c(1, -0.5)  # Coefficients for predictors
true_alpha <- 1.0       # Intercept

# Simulate count outcome
log_lambda <- X %*% true_beta + true_alpha
lambda <- exp(log_lambda)
y <- rpois(N, lambda)

# Prepare data for Stan
data_list <- list(
  N = N,
  K = K,
  X = X,
  y = y
)

# Define Stan model code
stan_model_code <- "
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
"

# Write the Stan model code to a file
writeLines(stan_model_code, "poisson_regression.stan")

# Compile the Stan model
stan_model <- stan_model(file = "poisson_regression.stan")

# Fit the model
fit <- sampling(stan_model, data = data_list, iter = 2000, chains = 4)

# Print the results
print(fit)

# Extract posterior samples
posterior_samples <- extract(fit)

# Convert to data frames
beta_df <- as.data.frame(posterior_samples$beta)
alpha_df <- as.data.frame(posterior_samples$alpha)

# Reshape for plotting
beta_long <- pivot_longer(beta_df, cols = everything(), names_to = "predictor", values_to = "value")
alpha_long <- alpha_df %>% rename(value = V1) %>% mutate(predictor = "Intercept")

# Plot posterior distributions
ggplot(beta_long, aes(x = value, fill = predictor)) +
  geom_density(alpha = 0.5) +
  labs(title = "Posterior Distributions of Coefficients", x = "Coefficient Value", y = "Density") +
  theme_minimal()

# Plot posterior distribution of alpha
ggplot(alpha_long, aes(x = value)) +
  geom_density(fill = "lightblue", alpha = 0.5) +
  labs(title = "Posterior Distribution of Intercept", x = "Intercept", y = "Density") +
  theme_minimal()


# zero inflated poisson ---------------------------------------------------

# Load required libraries
library(rstan)
library(ggplot2)
library(tidyr)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Simulate data
N <- 100  # Number of observations
K <- 2    # Number of predictors

# Simulate predictors (including intercept)
X <- cbind(1, matrix(rnorm(N * (K - 1)), ncol = K - 1))  # Add intercept as the first column

# True coefficients and parameters
true_beta <- c(1, -0.5)  # Coefficients for predictors
true_alpha <- 0.5       # Intercept for Poisson regression
true_pi <- 0.3          # Probability of zero inflation

# Simulate zero-inflation process
log_lambda <- X %*% true_beta + true_alpha
lambda <- exp(log_lambda)
inflation_prob <- rbinom(N, 1, true_pi)
y <- ifelse(inflation_prob == 1, 0, rpois(N, lambda))

# Prepare data for Stan
data_list <- list(
  N = N,
  K = K,
  X = X,
  y = y
)

# Define Stan model code
stan_model_code <- "
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
"

# Write the Stan model code to a file
writeLines(stan_model_code, "zero_inflated_poisson.stan")

# Compile the Stan model
stan_model <- stan_model(file = "zero_inflated_poisson.stan")

# Fit the model
fit <- sampling(stan_model, data = data_list, iter = 2000, chains = 4)

# Print the results
print(fit)

# Extract posterior samples
posterior_samples <- extract(fit)

# Convert to data frames
beta_df <- as.data.frame(posterior_samples$beta)
alpha_df <- as.data.frame(posterior_samples$alpha)
pi_df <- as.data.frame(posterior_samples$pi)

# Reshape for plotting
beta_long <- pivot_longer(beta_df, cols = everything(), names_to = "predictor", values_to = "value")
alpha_long <- alpha_df %>% rename(value = V1) %>% mutate(predictor = "Intercept")
pi_long <- pi_df %>% rename(value = V1) %>% mutate(predictor = "Zero Inflation")

# Plot posterior distributions
ggplot(beta_long, aes(x = value, fill = predictor)) +
  geom_density(alpha = 0.5) +
  labs(title = "Posterior Distributions of Coefficients", x = "Coefficient Value", y = "Density") +
  theme_minimal()

# Plot posterior distribution of alpha
ggplot(alpha_long, aes(x = value)) +
  geom_density(fill = "lightblue", alpha = 0.5) +
  labs(title = "Posterior Distribution of Intercept", x = "Intercept", y = "Density") +
  theme_minimal()

# Plot posterior distribution of zero inflation probability
ggplot(pi_long, aes(x = value)) +
  geom_density(fill = "lightgreen", alpha = 0.5) +
  labs(title = "Posterior Distribution of Zero Inflation Probability", x = "Zero Inflation Probability", y = "Density") +
  theme_minimal()


# negative binomial -------------------------------------------------------
# Load required libraries
library(rstan)
library(ggplot2)
library(tidyr)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Simulate data
N <- 100  # Number of observations
K <- 2    # Number of predictors

# Simulate predictors (including intercept)
X <- cbind(1, matrix(rnorm(N * (K - 1)), ncol = K - 1))  # Add intercept as the first column

# True coefficients and parameters
true_beta <- c(1, -0.5)  # Coefficients for predictors
true_alpha <- 0.5       # Intercept for Poisson component
true_phi <- 1.0         # Dispersion parameter

# Simulate count outcome
lambda <- exp(X %*% true_beta + true_alpha)
y <- rnbinom(N, size = true_phi, mu = lambda)

# Prepare data for Stan
data_list <- list(
  N = N,
  K = K,
  X = X,
  y = y
)

# Define Stan model code
stan_model_code <- "
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
"

# Write the Stan model code to a file
writeLines(stan_model_code, "negative_binomial_regression.stan")

# Compile the Stan model
stan_model <- stan_model(file = "negative_binomial_regression.stan")

# Fit the model
fit <- sampling(stan_model, data = data_list, iter = 2000, chains = 4)

# Print the results
print(fit)

# Extract posterior samples
posterior_samples <- extract(fit)

# Convert to data frames
beta_df <- as.data.frame(posterior_samples$beta)
alpha_df <- as.data.frame(posterior_samples$alpha)
phi_df <- as.data.frame(posterior_samples$phi)

# Reshape for plotting
beta_long <- pivot_longer(beta_df, cols = everything(), names_to = "predictor", values_to = "value")
alpha_long <- alpha_df %>% rename(value = V1) %>% mutate(predictor = "Intercept")
phi_long <- phi_df %>% rename(value = V1) %>% mutate(predictor = "Dispersion")

# Plot posterior distributions
ggplot(beta_long, aes(x = value, fill = predictor)) +
  geom_density(alpha = 0.5) +
  labs(title = "Posterior Distributions of Coefficients", x = "Coefficient Value", y = "Density") +
  theme_minimal()

# Plot posterior distribution of alpha
ggplot(alpha_long, aes(x = value)) +
  geom_density(fill = "lightblue", alpha = 0.5) +
  labs(title = "Posterior Distribution of Intercept", x = "Intercept", y = "Density") +
  theme_minimal()

# Plot posterior distribution of dispersion parameter
ggplot(phi_long, aes(x = value)) +
  geom_density(fill = "lightgreen", alpha = 0.5) +
  labs(title = "Posterior Distribution of Dispersion Parameter", x = "Dispersion Parameter", y = "Density") +
  theme_minimal()


# zero inflated negative binomial -----------------------------------------

# Load required libraries
library(rstan)
library(ggplot2)
library(tidyr)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Simulate data
N <- 100  # Number of observations
K <- 2    # Number of predictors

# Simulate predictors (including intercept)
X <- cbind(1, matrix(rnorm(N * (K - 1)), ncol = K - 1))  # Add intercept as the first column

# True coefficients and parameters
true_beta <- c(1, -0.5)  # Coefficients for predictors
true_alpha <- 0.5       # Intercept for Poisson component
true_phi <- 1.0         # Dispersion parameter for Negative Binomial
true_pi <- 0.3          # Probability of zero inflation

# Simulate zero-inflation process
log_lambda <- X %*% true_beta + true_alpha
lambda <- exp(log_lambda)
inflation_prob <- rbinom(N, 1, true_pi)
y <- ifelse(inflation_prob == 1, 0, rnbinom(N, size = true_phi, mu = lambda))

# Prepare data for Stan
data_list <- list(
  N = N,
  K = K,
  X = X,
  y = y
)

# Define Stan model code
stan_model_code <- "
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
"

# Write the Stan model code to a file
writeLines(stan_model_code, "zero_inflated_negative_binomial.stan")

# Compile the Stan model
stan_model <- stan_model(file = "zero_inflated_negative_binomial.stan")

# Fit the model
fit <- sampling(stan_model, data = data_list, iter = 2000, chains = 4, cores = 2)

# Print the results
print(fit)

# Extract posterior samples
posterior_samples <- extract(fit)

# Convert to data frames
beta_df <- as.data.frame(posterior_samples$beta)
alpha_df <- as.data.frame(posterior_samples$alpha)
phi_df <- as.data.frame(posterior_samples$phi)
pi_df <- as.data.frame(posterior_samples$pi)

# Reshape for plotting
beta_long <- pivot_longer(beta_df, cols = everything(), names_to = "predictor", values_to = "value")
alpha_long <- alpha_df %>% rename(value = V1) %>% mutate(predictor = "Intercept")
phi_long <- phi_df %>% rename(value = V1) %>% mutate(predictor = "Dispersion")
pi_long <- pi_df %>% rename(value = V1) %>% mutate(predictor = "Zero Inflation")

# Plot posterior distributions
ggplot(beta_long, aes(x = value, fill = predictor)) +
  geom_density(alpha = 0.5) +
  labs(title = "Posterior Distributions of Coefficients", x = "Coefficient Value", y = "Density") +
  theme_minimal()

# Plot posterior distribution of alpha
ggplot(alpha_long, aes(x = value)) +
  geom_density(fill = "lightblue", alpha = 0.5) +
  labs(title = "Posterior Distribution of Intercept", x = "Intercept", y = "Density") +
  theme_minimal()

# Plot posterior distribution of dispersion parameter
ggplot(phi_long, aes(x = value)) +
  geom_density(fill = "lightgreen", alpha = 0.5) +
  labs(title = "Posterior Distribution of Dispersion Parameter", x = "Dispersion Parameter", y = "Density") +
  theme_minimal()

# Plot posterior distribution of zero inflation probability
ggplot(pi_long, aes(x = value)) +
  geom_density(fill = "lightcoral", alpha = 0.5) +
  labs(title = "Posterior Distribution of Zero Inflation Probability", x = "Zero Inflation Probability", y = "Density") +
  theme_minimal()


# signal detection theory -------------------------------------------------

# Load required libraries
library(rstan)
library(ggplot2)
library(dplyr)
library(tidyr)

# Set seed for reproducibility
set.seed(123)

# Simulate data
N <- 100  # Number of trials
true_d_prime <- 1.5
true_c <- 0

# Calculate probabilities
p_hit <- 1 - pnorm(true_c + true_d_prime)  # p(hit)
p_fa <- pnorm(true_c)                      # p(false alarm)

# Simulate counts
hits <- rbinom(1, N, p_hit)
false_alarms <- rbinom(1, N, p_fa)
misses <- N - hits
correct_rejections <- N - false_alarms

# Prepare data for Stan
data_list <- list(
  N = N,
  hits = hits,
  misses = misses,
  false_alarms = false_alarms,
  correct_rejections = correct_rejections
)

# Define Stan model code
stan_model_code <- "
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
"

# Write the Stan model code to a file
writeLines(stan_model_code, "sdt_model.stan")

# Compile the Stan model
stan_model <- stan_model(file = "sdt_model.stan")

# Fit the model
fit <- sampling(stan_model, data = data_list, iter = 2000, chains = 4, cores = 2)

# Print the results
print(fit)

# Extract posterior samples
posterior_samples <- extract(fit)

# Convert to data frames
d_prime_df <- as.data.frame(posterior_samples$d_prime)
c_df <- as.data.frame(posterior_samples$c)

# Reshape for plotting
d_prime_long <- d_prime_df %>% rename(value = V1) %>% mutate(parameter = "d_prime")
c_long <- c_df %>% rename(value = V1) %>% mutate(parameter = "Criterion")

# Combine data for plotting
posterior_long <- bind_rows(d_prime_long, c_long)

# Plot posterior distributions
ggplot(posterior_long, aes(x = value, fill = parameter)) +
  geom_density(alpha = 0.5) +
  labs(title = "Posterior Distributions of SDT Parameters", x = "Parameter Value", y = "Density") +
  theme_minimal()

