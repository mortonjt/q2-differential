data {
  int<lower=0> N;         // number of samples
  int<lower=0> D;         // number of dimensions
  real slog[N];           // log normalization constants
  int M[N];               // binary group membership vector
  int y[N, D];            // observed microbe abundances
  real control_loc;
  real control_scale;
}

parameters {
  // parameters required for linear regression on the species means
  vector[D] control;
  vector[D] beta;
  matrix<lower=0>[2, D] alpha;
}

model {
  // setting priors ...
  to_vector(alpha) ~ lognormal(log(10), 1);
  control ~ normal(control_loc, control_scale);
  beta ~ normal(0, 5);
  // generating counts
  for (i in 1:N){
    vector[D] mu;
    mu = slog[i] + control + (M[i] - 1) * beta;
    y[i] ~ neg_binomial_2_log(mu, alpha[M[i]]);
  }
}

generated quantities {
  matrix[N, D] y_predict;
  matrix[N, D] log_lhood;
  for (n in 1:N){
    vector[D] mu;
    mu = slog[n] + control + (M[n] - 1) * beta;
    for (i in 1:D){
      y_predict[n, i] = neg_binomial_2_log_rng(mu[i], alpha[M[n], i]);
      log_lhood[n, i] = neg_binomial_2_log_lpmf(y[n, i] | mu[i], alpha[M[n], i]);
    }
  }
}
