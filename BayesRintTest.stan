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
  vector[2] spikeSlab;
  for(i in 1:Q)
    for(j in 1:Q)
      if(i==j)
        I[i,j]=1;
        else
        I[i,j]=0;
   spikeSlab[1]=0.005;
   spikeSlab[2]=1;
}
parameters{
  matrix[Px,Q] beta;
  vector<lower=0>[Q] sigma; //variance of likelihood, that is, sigma^2 in traditional notation
  real<lower=0> tau[Q]; //variance of components, that is, tau^2 in traditional notation
   simplex[K] pi[Q];
   real<lower=0,upper=1> chi;
    real<lower=0> lambda;
   // real<lower=0> lambda;
    real lcoef;
    
}
transformed parameters{
  real lp;
  matrix[4,Q] cVar;
   matrix[Q,Q] LAMBDAI;
  vector[2] lVar;
  
  cVar[,1] = tau[1]*components;
  cVar[,2] = tau[2]*components;
  lVar[1]= lambda .* spikeSlab[1];
  lVar[2]= lambda .* spikeSlab[2];
  {
    int index;
    matrix[4,Q] beta1;
    vector[2] lambda1;
    real accum;
    accum = 0;
    index=1;
    for(j in 1:Q){
      for(k in 1:Q){
        if(k==j){
            LAMBDAI[j,k]=1;
            if(k==Q)
              index=1;
        }
        else{
          if(j==2){
           accum=accum+log_mix(chi,normal_lpdf(lcoef|0,lVar[2]),normal_lpdf(lcoef|0,lVar[1]));
           LAMBDAI[j,k]=-1*lcoef;
          }
          else
           LAMBDAI[j,k]=0;
           index=index+1;
        }
      }
      for(i in 1:Px){
	      beta1[1,j] = log(pi[j,1]) + normal_lpdf(beta[i,j] | 0, cVar[1,j]);
	      beta1[2,j] = log(pi[j,2]) + normal_lpdf(beta[i,j] | 0, cVar[2,j]);
      	beta1[3,j] = log(pi[j,3]) + normal_lpdf(beta[i,j] | 0, cVar[3,j]);
	      beta1[4,j] = log(pi[j,4]) + normal_lpdf(beta[i,j] | 0, cVar[4,j]);
	      accum = accum + log_sum_exp(beta1[,j]);
      }
    }
    
    lp = accum;
  }
}

model{
  // matrix normal likelihood
  // we apply the trace trick: trace(A'B)=sum(A.*B)
  // additionally det(sigma*I)=sigma^dim(I), in our case would be sigma^Q 
//  lambda~inv_gamma(5,10);//this one works
  chi~ beta(1,1);
 // sigma[2]~normal(1,0.05);
  lambda ~ inv_gamma(10,100);
  target +=  -(N)*0.5*log(prod(sigma))-(0.5)*trace(crossprod(Y*LAMBDAI-X*beta)*diag_pre_multiply(1 ./ sigma,I));
  target += lp;//mixture contributions
}
generated quantities{
  matrix[N,Q] ypred;
  for(i in 1:N)
    for(j in 1:Q)
      ypred[i,j] = normal_rng(X[i,] * beta[,j], sigma[j]);
}
