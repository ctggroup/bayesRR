//UNPOOLED BayesR: individual component probabilities for each effect"
		data{ 
	int<lower=0> Px;
	int<lower=0> N;
	matrix[N,Px] X;
	real Y[N];
}
transformed data{
  vector[4] components;
  components[1]=1e-6;
  components[2]=1e-3;
  components[3]=1e-2;
  components[4]=1e-1;
}
parameters{
vector[Px] beta; // flat prior 

// this gives the lower bound for the variance components
real<lower=0> sigma;
// this is our vector of marker-specific variances
real<lower=0> tau;
vector[N] MU;
simplex[4] pi[Px];
}

transformed parameters{
real lp;
vector[4] cVar;
real sigmaS;
real tauS;
//sigmaS =1*sigma;
//tauS = 0.1*tau;
cVar=tau*components;

{
  vector[4] beta1; // flat prior 
  real accum;
  accum=0;
  for(i in 1:Px){ //mixture contributions to the joint distribution
    beta1[1]=log(pi[i,1])+normal_lpdf(beta[i]|0,cVar[1]);
    beta1[2]= log(pi[i,2])+normal_lpdf(beta[i]|0,cVar[2]);
    beta1[3]=log(pi[i,3])+normal_lpdf(beta[i]|0,cVar[3]);
    beta1[4]=log(pi[i,4])+normal_lpdf(beta[i]|0,cVar[4]);
    accum= accum+log_sum_exp(beta1);
  }
  lp=accum;
}
}
model{
tau ~ inv_gamma(2,1); //normal prior on variances as recommended in stan page
sigma ~ inv_gamma(2,1); // normal prior on variance as recommended in stan page
//MU ~ cauchy(0,1);   // fat tailed prior on the means
// the likelihood (vector expression)
Y ~ normal( X * beta, sigma);

target += lp; //mixture contribution to the joint distribution
}

generated quantities{
  vector[N] ypred;
  for(i in 1:N)
    ypred[i] = normal_rng(X[i,]*beta,sigma);
}
