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
  vector[D] eps;
  vector[2] delta;
  real a1;
  real a2;
  real sigma_e;
  real sigma_d;
}

model {
  // setting priors ...
  a1 ~ lognormal(log(10), 1);
  a2 ~ lognormal(log(0.1), 1);
  sigma_e ~ lognormal(log(0.1), 1);
  sigma_d ~ lognormal(log(0.1), 1);
  eps ~ lognormal(0, sigma_e);
  delta ~ lognormal(0, sigma_d);
  control ~ normal(control_loc, control_scale);
  beta ~ normal(0, 5);
  // generating counts
  for (i in 1:N){
    vector[D] mu;
    vector[D] alpha;
    mu = slog[i] + control + (M[i] - 1) * beta;
    alpha = (rep_vector(a1, D) ./ mu) + a2 + eps + delta[M[i]];
    y[i] ~ neg_binomial_2_log(mu, alpha[M[i]]);
  }
}

generated quantities {
  matrix[N, D] y_predict;
  matrix[N, D] log_lhood;
  for (n in 1:N){
    vector[D] mu;
    vector[D] alpha;
    mu = slog[n] + control + (M[n] - 1) * beta;
    alpha = (rep_vector(a1, D) ./ mu + a2) + eps + delta[M[n]];
    for (i in 1:D){
      y_predict[n, i] = neg_binomial_2_log_rng(mu[i], alpha[i]);
      log_lhood[n, i] = neg_binomial_2_log_lpmf(y[n, i] | mu[i], alpha[i]);
    }
  }
}
