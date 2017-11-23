//Pooled BayesR with phenotype interactions
//Y= X*B, being * matrix multiplication
//The model assumes pooled variance accross the columns of Y, this means that variance is the same for all columns
data{
  int<lower=0> Px;//dimension of the X covariates
  int<lower=0> N; //number of observations
  int<lower=0> Q; //dimension of phenotypic traits
  matrix[N,Px] X; //matrix of genetic covariates
  matrix[N,Q] Y; //matrix of phenotypic tratis
  int<lower=0> K; //number of components
  vector[K] components;
}
parameters{
  matrix[Px,Q] beta; //genetic effects
  real<lower=0> sigma; //variance of likelihood, that is, sigma^2 in traditional notation
  real<lower=0,upper=10> tau[Q]; //variance of components, that is, tau^2 in traditional notation
   simplex[K+1] pi;
}
transformed parameters{
  real lp;
  matrix[Q,K] cVar;
  
  {
    vector[K+1] beta1;
    real accum;
    accum = 0;
    // K+1 mixture of  normals as priors for the beta
    for(j in 1:Q){
      cVar[j,] = tau[j]*components';
      for(i in 1:Px){
        beta1[1] = log(pi[1]) + normal_lpdf(beta[i,j] | 0, 1e-4);
        for(k in 2:(K+1)){
          beta1[k] = log(pi[k]) + normal_lpdf(beta[i,j] | 0, cVar[j,k-1]);
        }
	      accum = accum + log_sum_exp(beta1);
      }
    }
    
    lp = accum;
  }
}

model{
  // matrix normal likelihood
  target += -(Q*N)*0.5*log(2*pi()*sigma)-(0.5/sigma)*sum((Y - X*beta) .* (Y - X*beta));
  target += lp;//mixture contributions
}
generated quantities{
  matrix[N,Q] ypred;
  for(i in 1:N)
    for(j in 1:Q)
      ypred[i,j] = normal_rng(X[i,] * beta[,j], sigma);
}
