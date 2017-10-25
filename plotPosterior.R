require(ggplot2)
require(rstan)
require(reshape2)
modelViolin <- function(stanFit,B,OLS,logB=FALSE){
  stan_effects <- extract(stanFit,"beta")$beta
  Bl<-B
  OLSl<-OLS
  coefNamesStan <- rownames(rstan::get_posterior_mean(stanFit,"beta"))
  coefNamesB<-list()
  N<-nrow(B)
  M<- ncol(B)
  accum=1
  
  for(i in 1:N)
    for(j in 1:M){
      coefNamesB[[accum]] <- paste("beta[",paste(
        toString(i),paste(
          ",",paste(
              toString(j),"]",sep=""
            ),sep=""
          ),sep=""
        ),sep=""
      )
      accum=accum+1
    }
  
  if(logB){
    Bl<-log10(abs(B))
    OLSl<-log10(abs(OLS))
    stan_effects<-log10(abs(stan_effects))  
  }
  staneffectsM<-stan_effects[,,1]
  for(i in 2:M){
    staneffectsM<-cbind(staneffectsM,stan_effects[,,i])
  }
  Bdf<-data.frame(parameter=unlist(coefNamesB),value=c(Bl))
  BOLSdf<-data.frame(parameter=unlist(coefNamesB),value=c(OLSl))
  colnames(staneffectsM)<-unlist(coefNamesB)
  staneffectsdf<-melt(staneffectsM,varnames = c("id","parameter"))
  p<-ggplot(staneffectsdf,aes(x=parameter,y=value))+geom_violin()+geom_point(data=Bdf,aes(x=parameter,y=value))+geom_point(data=BOLSdf,aes(x=parameter,y=value),shape=b)
  p
}
  