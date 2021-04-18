---
layout: post
comments: true
title: "Temporal Point Processes 2: Neural TPP models"
twitter_title: "Temporal Point Processes 2: Neural TPP models"
subtitle: "How can we define flexible TPP models using neural networks?"
date: 2021-03-01 09:00:00 +0200
background: '/img/posts/tpp2/header.jpg'
image: '/img/posts/tpp2/header.jpg'
---

## TL;DR
- *TLDR point 1*


In the [previous blog post](https://shchur.github.io/2020/12/17/tpp1-conditional-intensity.html), we learned about different ways to describe a temporal point process (TPP) --- a generative model for variable-length event sequences in continuous time.
We have also seen some [examples of TPPs](https://shchur.github.io/2020/12/17/tpp1-conditional-intensity.html#defining-tpps-using-the-conditional-intensity-function), such as the Poisson process (that captures a global trend in the events frequency) and the Hawkes process (that captures self-exciting / bursty event occurrences). 

In this post, we will see how to define more flexible TPPs based on neural networks that can capture more complex dependencies between events.
In the process, we will
- Learn how to parametrize TPPs using neural networks;
- Derive the likelihood function for TPPs (that we will use as the training objective for our model);
- Implement a neural TPP model in PyTorch.

<!-- 1. We will derive the likelihood function for TPPs that we will use as the training objective for our model.
2. We will discuss the building blocks of a neural TPP.
3. We will implement such a neural TPP model in PyTorch. -->

If you haven't read the [previous post in the series](https://shchur.github.io/2020/12/17/tpp1-conditional-intensity.html), I highly recommend checking it out to get familiar with the main concepts and notation.
Alternatively, click on the arrow below to see a short summary.
<details>
  <summary markdown='span'>Summary</summary>
  We denote 
</details>

&nbsp;

## Constructing a neural TPP model

As discussed last time, we can define a TPP autoregressively by specifying the distribution $$P_i^*(t_i) := P_i^*(t_i \vert \mathcal{H}_{t_i})$$ of the next arrival time $$t_i$$ given the history of past events $$\mathcal{H}_{t_i} = \{t_1, \dots, t_{i-1}\}$$.
For convenience, we will instead consider the equivalent representation in terms of the *inter-event* times $$\tau_i := t_i - t_{i-1}$$.
That is, we will model the distribution $$P_i^*(\tau_i)$$ of the next inter-event time $$\tau_i$$ given the past events $$\mathcal{H}_{t_i}$$.

FIGURE: Inter-event and arrival times.

How can we parametrize the conditional distribution $$P_i^*(\tau_i)$$ with a neural network?
A simple and elegant answer to this question was proposed in the seminal work by [[Du, Dai, Trivedi, Gomez-Rodriguez and Song, 2016]](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf).

1. Encode the event history $$\mathcal{H}_{t_i} = \{t_1, \dots, t_{i-1}\}$$ into a *fixed-dimensional* context vector $$\boldsymbol{c}_i \in \mathbb{R}^d$$ using a neural network.
2. Pick a parametric probability density function $$f(\cdot \vert \boldsymbol{\theta})$$ that defines the distribution of a non-negative random variable (e.g., PDF of the [<ins>exponential distribution</ins>](https://en.wikipedia.org/wiki/Exponential_distribution) or [<ins>Weibull distribution</ins>](https://en.wikipedia.org/wiki/Weibull_distribution)).
3. Use the context vector $$\boldsymbol{c}_i$$ to obtain the parameters $$\boldsymbol{\theta}_i$$. Plug in $$\boldsymbol{\theta}_i$$ into $$f(\cdot \vert \boldsymbol{\theta})$$ to obtain the PDF $$f(\tau_i \vert \boldsymbol{\theta}_i)$$ of the conditional distribution $$P_i^*(\tau_i)$$.

FIGURE: Schematic representation of a neural TPP.

We will now look at each of these steps in more detail.

## Likelihood function of a TPP
Log-likelihood is the default training objective for generative probabilistic models, and TPPs are no exception.
Suppose we have observed a single event with arrival time $$t_1$$ in the time interval $$[0, T]$$.

$$
\begin{aligned}
p(\{t_1\}) =& \Pr(\text{first event in $[t_1, t_1 + dt)$}) \cdot \Pr(\text{second event after $T$} \mid t_1)\\
=& f_1^*(t_1) dt \cdot S_2^*(T)
\end{aligned}
$$

Using the same reasoning, we can derive the likelihood for a sequence $$\boldsymbol{t} = \{t_1, t_2, \dots, t_N\}$$ consisting of $$N$$ events as

$$
\begin{aligned}
p(\boldsymbol{t}) =& \left(\prod_{i=1}^{N} f_i^*(t_i)\right) S_{N+1}^*(T)
\end{aligned}
$$

by definition of the hazard function $$h_i^*(t)$$

$$
\begin{aligned}
p(\boldsymbol{t}) =& \left(\prod_{i=1}^{N} f_i^*(t_i)\right) S_{N+1}^*(T)\\
=& \left(\prod_{i=1}^{N} h_i^*(t_i) S_i^*(t_i)\right) S_{N+1}^*(T)\\
=& \left(\prod_{i=1}^{N} h_i^*(t_i)\right) \left(\prod_{i=1}^{N+1} S_i^*(t_i)\right)
\end{aligned}
$$

where we defined $$t_{N+1}=T$$.

$$S_i^*(t) = \exp \left(-\int_{t_{i-1}}^{t} h_i^*(u) du\right)$$

Finally, we obtain the log-likelihood by applying the logarithm

$$
\begin{aligned}
\log p(\boldsymbol{t}) =& \sum_{i=1}^{N} \log h_i^*(t_i) - \sum_{i=1}^{N+1} \left(\int_{t_{i-1}}^{t_i} h_i^*(u) du\right)
\end{aligned}
$$

<details>
  <summary markdown='span'>Collapsed Block</summary>
    $$
    \begin{aligned}
    \log p(\boldsymbol{t}) =& \sum_{i=1}^{N} \log \lambda^*(t_i) - \sum_{i=1}^{N+1} \left(\int_{t_{i-1}}^{t_i} \lambda^*(u) du\right)\\
    =& \sum_{i=1}^{N} \log \lambda^*(t_i) - \int_{0}^{T} \lambda^*(u) du
    \end{aligned}
    $$
</details>

Can also be expressed in terms of the conditional intensity function $$\lambda^*(t)$$.

This is the formula that you may see in many papers in textbooks.
However, in reality, when actually implementing TPP models (especially neural ones), we nearly always work with the hazard functions $$h_i^*$$, so I will stick to this formulation of the log-likelihood.


How do we actually parametrize the conditional hazard function $$h_i^*$$ for *each* event $$i$$?



```python
def function():
    print('Yes')
```

## Encoding the event history into a vector
Tada


## Parametrizing the conditional hazard function
Tada


## References

- Reference 1


## Footnotes

[^1]: 
    What if I put this here?
