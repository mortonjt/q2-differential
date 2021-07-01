data {
  int<lower=0> N;       // number of samples
  int<lower=0> D;       // number of dimensions
  int depth[N];         // sequencing depths of microbes
  int[N] M;             // binary group membership vector
  int y[N, D];          // observed microbe abundances
}

parameters {
  // parameters required for linear regression on the species means
  matrix[2, D] beta;
}

transformed parameters {
  vector[D] prior = rep_vector(0.5, D);
}

model {
  // setting priors ...
  beta[1] ~ dirichlet(prior);
  beta[2] ~ dirichlet(prior);
  // generating counts
  for (n in 1:N){
    target += multinomial_lpmf(y[n,] | to_vector(beta[M[n],]));
  }
}

generated quantities {
  matrix[N, D] y_predict;
  matrix[N, D] p_predict;
  vector[D] logfold_diff;
  vector[N] log_lhood;
  for (n in 1:N){
    p_predict[n,] = dirichlet_rng(to_vector(beta[M[n],]));
    y_predict[n,] = multinomial_rng(to_vector(beta[M[n],]), depth[n]);
    log_lhood[n] = multinomial_lpmf(y[n,] | to_vector(to_vector(beta[M[n],])));
  }
  logfold_diff = log(beta[1] / beta[2]);
}
