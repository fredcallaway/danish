---
title:  Effect of parameters on net performance
output: html_document
---



#### Abstract
This report summarizes the effect of network parameters on word segmentation and ALL accuracy. We find that network parameters have a substanital effect on both of these measures. For word segmentation, this effect is generally not strong enough to put our initial finding of English superiority into doubt (with the exception of an extreme momentum value of 0.98). However, for the ALL experiment, the strong effect of network parameters in combination with initially weak findings make our initial findings of danish superiority questionable.

---------

## How do net parameters affect word segmentation?
First, let's look at average word segmentation performance.

_Note: For all graphs, error bars indicate the standard error of the mean._

![plot of chunk average-segmentation](figs/average-segmentation-1.png) 

In concordance with our previous results, English does better. However the difference is not as pronounced. __English word F has lowered from 4.5 to 3.5.__

### Individual effects
Of course, this does not tell us much. There could be important effects that cancel each other out. Let's look at an ANOVA for all the parameters we varied.


```
##              Df Sum Sq Mean Sq F value   Pr(>F)    
## language      1 0.1871  0.1871  25.095  1.5e-06 ***
## rate          2 0.1239  0.0619   8.306 0.000377 ***
## momentum      2 0.7572  0.3786  50.769  < 2e-16 ***
## rand_range    2 0.0084  0.0042   0.561 0.572040    
## hidden        2 0.0004  0.0002   0.030 0.970885    
## Residuals   152 1.1335  0.0075                     
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

We find that __language__, __rate__, and __momentum__ are all highly significant. However, these differences only really matter to us if they interact with language. Such an interaction could call into question our initial finding that English nets performed better on the segmentation task than Danish nets. Let's look for such an interaction:

_Note: we do not include interactions with number of hiddent units here because the excess of comparisons prevents R from giving significance. A separate analysis indicated that this parameter does not significantly interact with the others._

<!-- 

![plot of chunk segmentation-rate](figs/segmentation-rate-1.png) 
 -->


```
##                                    Df Sum Sq Mean Sq F value   Pr(>F)    
## language                            1 0.1871  0.1871  65.303 1.10e-12 ***
## rate                                2 0.1239  0.0619  21.615 1.34e-08 ***
## momentum                            2 0.7572  0.3786 132.111  < 2e-16 ***
## rand_range                          2 0.0084  0.0042   1.459    0.237    
## hidden                              2 0.0004  0.0002   0.077    0.926    
## language:rate                       2 0.0108  0.0054   1.892    0.156    
## language:momentum                   2 0.3837  0.1919  66.949  < 2e-16 ***
## rate:momentum                       4 0.3404  0.0851  29.699  < 2e-16 ***
## language:rand_range                 2 0.0002  0.0001   0.034    0.967    
## rate:rand_range                     4 0.0062  0.0015   0.541    0.706    
## momentum:rand_range                 4 0.0030  0.0008   0.262    0.902    
## language:rate:momentum              4 0.0104  0.0026   0.908    0.462    
## language:rate:rand_range            4 0.0120  0.0030   1.043    0.389    
## language:momentum:rand_range        4 0.0121  0.0030   1.059    0.380    
## rate:momentum:rand_range            8 0.0122  0.0015   0.530    0.832    
## language:rate:momentum:rand_range   8 0.0387  0.0048   1.687    0.110    
## Residuals                         106 0.3038  0.0029                     
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

As we see, there is a significant __language by momentum__ interaction. Let's see what this looks like:

![plot of chunk segmentation-differences](figs/segmentation-differences-1.png) 

Observing this graph, we see that __the pattern of English nets performing better than Danish nets generally holds.__ There is an exception for 0.98, however this is a very high value, so erratic behavior may not be a concern.

There was additionally a rate by momentum interaction. Althougth there was no significant three-way interaction with language, let's see what this looks like.

![plot of chunk segmentation-lang-rate-momentum](figs/segmentation-lang-rate-momentum-1.png) 

It appears that this interaction is only noticable at momentum = 0.98.

## How do net parameters affect ALL performance?

Once again, we'll start by averaging across all parameters to see if the general pattern of results is concordant with our initial findings.

![plot of chunk unnamed-chunk-3](figs/unnamed-chunk-3-1.png) 

As with the segmentation results, we find that the general pattern holds, but with less strength. Notably, __performance of Danish nets on the contoid language has dropped from ~.72 to ~.68.__ (Compare to earlier figure). This difference is concerning because our initial results were only marginally significant. Let's check to see if we still have significance.


```
##                     Df Sum Sq  Mean Sq F value Pr(>F)  
## language             1 0.0157 0.015736   2.901 0.0895 .
## condition            1 0.0098 0.009757   1.799 0.1808  
## language:condition   1 0.0295 0.029469   5.432 0.0204 *
## Residuals          320 1.7360 0.005425                 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

There is a significant __language by condition__ interaction and a marginally significant effect of language alone. Looking at the graph, these differences likely come down to the reduced performance of English nets on the contoid language. As with our initial findings, the pattern of contoid vs. vocoid is opposite what we found for human subjects.

### Individual effects
Now let's look at which parameters have an effect on ALL performance in each task. We'll start by looking at accuracy on the contoid language because of its difference here from our initial findings.

#### Contoids


```
##                                    Df Sum Sq Mean Sq F value   Pr(>F)    
## language                            1 0.0441  0.0441  56.732 1.75e-11 ***
## rate                                2 0.0668  0.0334  42.948 2.18e-14 ***
## momentum                            2 0.6780  0.3390 435.762  < 2e-16 ***
## rand_range                          2 0.0046  0.0023   2.946   0.0569 .  
## hidden                              2 0.0002  0.0001   0.100   0.9049    
## language:rate                       2 0.0069  0.0035   4.435   0.0141 *  
## language:momentum                   2 0.0192  0.0096  12.371 1.48e-05 ***
## rate:momentum                       4 0.0019  0.0005   0.616   0.6520    
## language:rand_range                 2 0.0000  0.0000   0.014   0.9863    
## rate:rand_range                     4 0.0046  0.0011   1.463   0.2186    
## momentum:rand_range                 4 0.0073  0.0018   2.357   0.0582 .  
## language:rate:momentum              4 0.0077  0.0019   2.461   0.0498 *  
## language:rate:rand_range            4 0.0036  0.0009   1.153   0.3359    
## language:momentum:rand_range        4 0.0050  0.0013   1.609   0.1774    
## rate:momentum:rand_range            8 0.0033  0.0004   0.534   0.8287    
## language:rate:momentum:rand_range   8 0.0070  0.0009   1.120   0.3558    
## Residuals                         106 0.0825  0.0008                     
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

It appears that there is significant interaction between __language__, __rate__, and __momentum__. Let's look at how these parameters effect performance on the contoid experiment for each language.

![plot of chunk contoids](figs/contoids-1.png) 

It appears that our initial choice of parameters (0.1 and 0.95) was partially responsible for the initial difference we saw in performance on contoids by each language. Let's expand this analysis to look at the relationship between contoid and vocoid performance.

#### Contoids and vocoids
We'll start with an ANOVA of contoid_accuracy - vocoid_accuracy.


```
##                                    Df  Sum Sq Mean Sq F value   Pr(>F)    
## language                            1 0.05894 0.05894  36.799 2.05e-08 ***
## rate                                2 0.00236 0.00118   0.737  0.48079    
## momentum                            2 0.00365 0.00182   1.139  0.32395    
## rand_range                          2 0.00513 0.00256   1.601  0.20653    
## hidden                              2 0.00013 0.00006   0.040  0.96047    
## language:rate                       2 0.01972 0.00986   6.158  0.00295 ** 
## language:momentum                   2 0.00163 0.00082   0.510  0.60197    
## rate:momentum                       4 0.00225 0.00056   0.351  0.84271    
## language:rand_range                 2 0.00264 0.00132   0.823  0.44169    
## rate:rand_range                     4 0.00528 0.00132   0.824  0.51269    
## momentum:rand_range                 4 0.00932 0.00233   1.454  0.22138    
## language:rate:momentum              4 0.00851 0.00213   1.328  0.26445    
## language:rate:rand_range            4 0.00646 0.00161   1.008  0.40693    
## language:momentum:rand_range        4 0.00630 0.00157   0.983  0.41993    
## rate:momentum:rand_range            8 0.00557 0.00070   0.435  0.89768    
## language:rate:momentum:rand_range   8 0.01667 0.00208   1.301  0.25111    
## Residuals                         106 0.16977 0.00160                     
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

We see that language by rate is significant.

![plot of chunk contoid-vocoid](figs/contoid-vocoid-1.png) 

Again, we see that rate and momentum have a substantial impact on our intitial qualitative findings. Although on average we see a pattern of danish superiority, this pattern does not hold for several conditions. The pattern of english being worse on the contoids and danish being better on the contoids is more stable, but still not entirely so. Thus, we cannot conclude that our initial findings for ALL performance are stable as we vary network parameters. 
