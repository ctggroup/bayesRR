//POOLED BayesR: component probabilities for all effects"
data{
  int<lower=0> Px;
  int<lower=0> N;
  int K;
  matrix[N,Px] X;
  real Y[N];
  vector[K] components; //WITHOUT THE SPIKE AT ZERO COMPONENT
}

parameters{
  vector[Px] beta; // flat prior
  // this gives the lower bound for the variance components
  real<lower=0> sigma;
  // this is our vector of marker-specific variances
  real<lower=0,upper=10> tau;
  simplex[K+1] pi;
}

transformed parameters{
  real lp;
  vector[K] cVar;
  cVar=tau*components;
  {
    vector[K+1] beta1; // flat prior
    real accum;
    accum=0;
    for(i in 1:Px){ //mixture contributions to the joint distribution
      beta1[1]=log(pi[1])+normal_lpdf(beta[i]|0,1e-4); //SPIKE AT ZERO
      for(k in 1:K){
        beta1[k+1]= log(pi[k+1])+normal_lpdf(beta[i]|0,cVar[k]);
      }
      accum= accum+log_sum_exp(beta1);
    }
    lp=accum;
  }
}
model{
  sigma ~ inv_gamma(2,1); 
  mu ~ cauchy(0,1);
  Y ~ normal(mu + X * beta, sigma);
  target += lp; //mixture contribution to the joint distribution
}

