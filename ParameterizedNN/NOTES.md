# NOTES on the Seminar

## 09.11.22

Find out likelihood
<br>
$l = \frac{p(x_0|\theta)}{p(x_0|theta_0)}$
<br>
$s = \frac{1}{1+exp^{-l}}$
<br>

Minimizing the negative loglikelihood

$\theta = argmax (loglikelihood - C)$

Keep one x fixed and variate theta. For 1 and 2 dimensions.


Take some x (5 points)data from a gaussian  -> get llhod from model (sum over likelihood)
and plot the over the theta space.

The more x points we take the narrower the parabel will be and approximate to true value.

Alternative (not that many points):


Iterate may times and then plot a histrogram of the minima -> should convert to the true value.

### bayesian analysis with prior

1. log probability of bayesian formular
2. multiply by one in the form of $p(x|\theta_0)$
#. Give up calculating the evidence but sample from prior

### explanation of mcmc
???
