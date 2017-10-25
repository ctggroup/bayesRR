//Pooled BayesR with phenotype interactions
//Y= X*B+Y*LAMBDA, being * matrix multiplication
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
transformed data{
  matrix[Q,Q] I;
  matrix[Q*Q-Q,2] spikeSlab;
  for(i in 1:Q)
    for(j in 1:Q)
      if(i==j)
        I[i,j]=1;
        else
        I[i,j]=0;
   spikeSlab[,1]=rep_vector(1e-6,Q*Q-Q);
   spikeSlab[,2]=rep_vector(1,Q*Q-Q);
}
parameters{
  matrix[Px,Q] beta;
  vector<lower=0.7,upper=1.0>[Q] sigma; //variance of likelihood, that is, sigma^2 in traditional notation
  real<lower=0,upper=10> tau; //variance of components, that is, tau^2 in traditional notation
   simplex[K] pi;
   real<lower=0,upper=1> chi[Q*Q-Q];
    vector<lower=0,upper=10>[Q*Q-Q] lambda;
}
transformed parameters{
  real lp;
  vector[4] cVar;
  matrix[Q,Q] LAMBDAI;
  matrix[Q*Q-Q,2] lVar;
  matrix[Q,Q] LAMBDAIinv;
  
  cVar = tau*components;
  lVar[,1]= lambda .* spikeSlab[,1];
  lVar[,2]= lambda .* spikeSlab[,2];
  {
    int index;
    vector[4] beta1;
    vector[2] lambda1;
    real accum;
    accum = 0;
    index=1;
    for(j in 1:Q){
      for(k in 1:Q){
        if(k==j)
            LAMBDAI[j,k]=1;
        else{
           accum=log_mix(chi[index],normal_lpdf(LAMBDAI[j,k]|0,lVar[index,2]),normal_lpdf(LAMBDAI[j,k]|0,lVar[index,1]));
           index=index+1;
        }
      }
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
  LAMBDAIinv=inverse(LAMBDAI);
}

model{
  // matrix normal likelihood
  // we apply the trace trick: trace(A'B)=sum(A.*B)
  // additionally det(sigma*I)=sigma^dim(I), in our case would be sigma^Q 
  target += -(N)*0.5*log(2*pi()*prod(sigma))-(0.5)*trace(crossprod(Y-X*beta*LAMBDAIinv)*((LAMBDAIinv*diag_pre_multiply(1 ./ sigma,I))*(LAMBDAIinv')));
  target += lp;//mixture contributions
}
generated quantities{
  matrix[N,Q] ypred;
  for(i in 1:N)
    for(j in 1:Q)
      ypred[i,j] = normal_rng(X[i,] * beta[,j], sigma[j]);
}
