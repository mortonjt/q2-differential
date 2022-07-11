data {
  int<lower=0> N;         // number of samples
  int<lower=0> D;         // number of dimensions
  real slog[N];           // log normalization constants
  int M[N];               // binary group membership vector
  int y[N];            // observed microbe abundances
  real control_loc;
  real control_scale;
}

parameters {
  // parameters required for linear regression on the species means
  real control;
  real beta;
  vector<lower=0>[2] alpha;
}

model {
  // setting priors ...
  to_vector(alpha) ~ lognormal(log(10), 1);
  control ~ normal(control_loc, control_scale);
  beta ~ normal(0, 5);
  // generating counts
  for (i in 1:N){
    real mu;
    mu = slog[i] + log_inv_logit(control + (M[i] - 1) * beta);
    y[i] ~ neg_binomial_2_log(mu, inv(alpha[M[i]]));
  }
}

generated quantities {
  vector[N] y_predict;
  vector[N] log_lhood;
  for (n in 1:N){
    real mu;
    mu = slog[n] + log_inv_logit(control + (M[n] - 1) * beta);
    y_predict[n] = neg_binomial_2_log_rng(mu, inv(alpha[M[n]]));
    log_lhood[n] = neg_binomial_2_log_lpmf(y[n] | mu, inv(alpha[M[n]]));
  }
}
