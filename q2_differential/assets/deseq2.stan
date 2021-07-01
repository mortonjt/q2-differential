data {
  int<lower=0> N;         // number of samples
  int<lower=0> D;         // number of dimensions
  real slog[N];           // log normalization constants
  int[N] M;               // binary group membership vector
  int y[N, D];            // observed microbe abundances
  real beta_s;            // prior for beta scale
  real alpha_s;           // scale for dispersion normal prior
}

parameters {
  // parameters required for linear regression on the species means
  vector[D] beta_int;
  vector[D] beta_diff;
  vector<lower=0>[D] alpha;
}

transformed parameters {
  matrix[N, D] qlog;
  for (i in 1:N){
    qlog[i] = beta_int + (M[n] - 1) * beta_diff;
  }
}

model {
  // setting priors ...
  // the dispersion trend function is too complicated
  // let's just set a hyper-prior over alpha_m
  alpha_m ~ normal(0, 1);
  alpha ~ lognormal(alpha_m., alpha_s);
  beta ~ normal(0, beta_s);
  // generating counts
  for (i in 1:N){
    y[i] ~ neg_binomial_2_log(slog[i] + qlog[i], alpha[i]);
  }
}

generated quantities {
  matrix[N, D] y_predict;
  matrix[N, D] log_lhood;

  for (n in 1:N){
    for (i in 1:D){
      y_predict[n, i] = neg_binomial_2_log_rng(slog[n] + qlog[n], alpha[i]);
      log_lhood[n, i] = neg_binomial_2_log_lpmf(y[n, i] | slog[n] + qlog[n], alpha[i]);
    }
  }
}
o
