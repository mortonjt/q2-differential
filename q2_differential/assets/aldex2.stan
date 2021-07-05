data {
  int<lower=0> N;       // number of samples
  int<lower=0> D;       // number of dimensions
  int depth[N];         // sequencing depths of microbes
  int M[N];             // binary group membership vector
  int y[N, D];          // observed microbe abundances
}

parameters {
  // parameters required for linear regression on the species means
  simplex[D] beta[2];
}

transformed parameters {
  vector<lower=0>[D] prior = rep_vector(0.5, D);
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
  int y_predict[N, D];
  vector[D] p_predict[N];
  vector[N] log_lhood;
  for (n in 1:N){
    p_predict[n,] = dirichlet_rng(to_vector(beta[M[n],]));
    y_predict[n,] = multinomial_rng(to_vector(beta[M[n],]), depth[n]);
    log_lhood[n] = multinomial_lpmf(y[n,] | to_vector(to_vector(beta[M[n],])));
  }
}
