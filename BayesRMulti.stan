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
  matrix[Px,Q] beta;
  real<lower=0> sigma; //variance of likelihood, that is, sigma^2 in traditional notation
  real<lower=0,upper=10> tau; //variance of components, that is, tau^2 in traditional notation
   simplex[K] pi;
}
transformed parameters{
  real lp;
  vector[4] cVar;
  cVar = tau*components;
  {
    vector[4] beta1;
    vector[2] lambda1;
    real accum;
    accum = 0;
    for(j in 1:Q){
      for(i in 1:Px){
	beta1[1] = log(pi[1]) + normal_lpdf(beta[i,j] | 0, cVar[1]);
	beta1[2] = log(pi[2]) + normal_lpdf(beta[i,j] | 0, cVar[2]);
	beta1[3] = log(pi[3]) + normal_lpdf(beta[i,j] | 0, cVar[3]);
	beta1[4] = log(pi[4]) + normal_lpdf(beta[i,j] | 0, cVar[4]);
	accum = accum + log_sum_exp(beta1);
      }
    }
    
    lp = accum;
  }
}

model{
  // matrix normal likelihood
  // we apply the trace trick: trace(A'B)=sum(A.*B)
  // additionally det(sigma*I)=sigma^dim(I), in our case would be sigma^Q 
  target += -(Q*N)*0.5*log(2*pi()*sigma)-(0.5/sigma)*sum((Y - X*beta) .* (Y - X*beta));
  target += lp;//mixture contributions
}
generated quantities{
  matrix[N,Q] ypred;
  for(i in 1:N)
    for(j in 1:Q)
      ypred[i,j] = normal_rng(X[i,] * beta[,j], sigma);
}
