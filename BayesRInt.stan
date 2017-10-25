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
  real CHIV;
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
   spikeSlab[,1]=rep_vector(0.000000005,Q*Q-Q);
   spikeSlab[,2]=rep_vector(1,Q*Q-Q);
   
}
parameters{
  matrix[Px,Q] beta;
  vector<lower=0>[Q] sigma; //variance of likelihood, that is, sigma^2 in traditional notation
  real<lower=0> tau[Q]; //variance of components, that is, tau^2 in traditional notation
   simplex[K] pi[Q];
   //real<lower=0,upper=1> chi[Q*Q-Q];
    vector<lower=0>[Q*Q-Q] lambda;
   // real<lower=0> lambda;
    vector[Q*Q-Q] lcoef;
    real<lower=0> chi;
}
transformed parameters{
  real lp;
  matrix[4,Q] cVar;
   matrix[Q,Q] LAMBDAI;
  vector[Q*Q-Q] lVar;
  
  cVar[,1] = tau[1]*components;
  cVar[,2] = tau[2]*components;
  //lVar[,1]= lambda .* spikeSlab[,1];
  //lVar[,2]= lambda .* spikeSlab[,2];
  lVar= chi*lambda;
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
              if(j==Q)
                index=1;
        }
        else{
           LAMBDAI[j,k]=-1*lcoef[index];
           index=index+1;
        }
      }
    }
    for(j in 1:Q){
      
      
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
  //chi[1]~ beta(0.1,1000);
  //chi[2]~ beta(1000,0.1);
 // sigma[2]~normal(1,0.05);
  chi~cauchy(0,CHIV);
  lcoef~ normal(0,lVar);
  lambda~cauchy(0,1);
 // lambda ~ inv_gamma(10,100);
  target += lp;//mixture contributions
  target +=  -(N)*0.5*log(prod(sigma))-(0.5)*trace(crossprod(Y*LAMBDAI-X*beta)*diag_pre_multiply(1 ./ sigma,I));

}
generated quantities{
  matrix[N,Q] ypred;
  for(i in 1:N)
    for(j in 1:Q)
      ypred[i,j] = normal_rng(X[i,] * beta[,j], sigma[j]);
}
