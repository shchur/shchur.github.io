---
layout: post
comments: true
title: "Temporal Point Processes 1: The Conditional Intensity Function"
subtitle: "How can we define generative models for variable-length event sequences?"
date: 2020-09-01 09:00:00 +0200
background: '/img/posts/1.jpg'
---

## What is a point process?
Probabilistic generative models are the bread and butter of modern machine learning.
They allow us to make predictions, find anomalies and learn useful representations of the data.
Most of the time, applying the generative model involves learning the probability distribution $$p(\bm{x})$$ over our data points $$\bm{x}$$.

We know what to do if $$\bm{x}$$ is a vector in $$\mathbb{R}^D$$
--- simply use a multivariate Gaussian or, if we need something more flexible, our favorite normalizing flow model.
But what if a single realization of our probabilistic model corresponds to a *set* of vectors $$\{\bm{x}_1, ..., \bm{x}_N\}$$?
Even worse, what if both $$N$$, the number of the vectors, as well as their locations $$\bm{x}_i$$ are random?
This is not some hypothetical scenario --- processes generating such data are abundant in the real world: 
- Transactions generated each day in a financial system
- Locations of disease outbreaks in a city, recorded each week
- Times and locations of earthquakes in some geographic region within a year

Point processes provide a framework for modeling and analyzing such data.
Intuitively, we can think of a point process as a probability distribution over variable-sized sets of objects.
Each realization of a point process is a set $$\{\bm{x}_1, \dots, \bm{x}_N\}$$ consisting of a random number $$N$$ of *points* $$\bm{x}_i$$ that live in some space $$\mathcal{X}$$, hence the name "point process".
Depending on the choice of the space $$\mathcal{X}$$, we distinguish among different types of point processes.
For example, $$\mathcal{X} \subseteq \mathbb{R}^D$$ corresponds to a so-called *spatial point process*, where every point $$\bm{x}_i$$ can be viewed as a random location in space (e.g., a location of a disease outbreak).


Figure: Two realizations of a spatial point process on $$\mathbb{R}^2$$.


Another important case, to which I will dedicate the rest of this post (and, hopefully, several future ones), are *temporal point processes* (TPPs), defined on the real half-line $$\mathcal{X} \subseteq [0, \infty)$$.
We can interpret the points in a TPP as events happening in continuous time, and therefore usually denote them as $$t_i$$ (instead of $$\bm{x}_i$$).

Figure: Two realizations of a temporal point process on $$[0, T]$$.

TPPs are interesting 

intermittent demand forecasting



Temporal point processes (TPP) have a nice structure to them .... 
the natural ordering of the events imposed by $$\mathbb{R}$$ makes it easy to describe the distribution ... 
... this ... which makes most of them 



Temporal point processes are a special case, where the points are located in $$\mathbb{R}_+$$, the positive real line.
We can think of $$\mathbb{R}_+$$ as representing the time. 
The points in a ... 
and of the points $$t_1, t_2, ... \in \mathbb{R}_+$$ as a sequence of events that happen one after another.



The theory of temporal point processes was mostly developed near the middle of the 20th century, taking roots in measure theory and stochastic processes.
For this reason, the notation and jargon used in TPP literature may sound strange and unfamiliar to people with a machine learning background.
In reality, though, most TPP-related concepts can be easily translated into the familiar language of probabilistic machine learning.
In this series of blog posts I will try to ... the two perspectives


The fundamental concept that you will find in virtually every TPP paper or textbook is the *conditional intensity function*, commonly denoted as $$\lambda^*(t)$$.
... we can view it from two angles.


### The machine learning perspective

In machine learning, we usually characterize continuous probability distributions by specifying their probability density functions (PDF).

Probability density function $$p_i(t_i \vert \mathcal{H}_{t_{i}})$$.

However, there exist other options that might be more useful in certain contexts.

What is the probability that the next event $$t_i$$ will happen in the interval $$[t, t + dt)$$?

This is not the only way to characterize this distribution.
For example, we could specify its cumulative distribution function (CDF)
$$F_i(t_i \vert \mathcal{H}_{t_{i}})$$

Survival function (SF)

Hazard function


Survival analysis.


<!-- Most of the theory was developed in the 60s-70s, ... roots in measure theory so the jargon ... might seem very intimidating at first. They had different goals - establishing theoretical foundations. Now we are facing different challenges - we should change the way we think about TPPs... . 

I this first blog post I will ... that I personally find more intuitive.

Might seem intimidating at first, with lots of jargon, but turns out to be ... much more simple

- Loaded with jargon
- Information is spread across many sources
- most existing resources either don't go deep enough or require the reader to have a PhD in measure theory.

The goal of this series of blog posts is to show you that 

In this series of blog posts I will try to summarize what I learned ...  -->

<!-- ### TL;DR -->

<!-- - Temporal point processes are basically autoregressive probabilistic models that define the distribution of discrete events in continuous time.
- Conditional intensity is one of many equivalent ways to specify a distribution over the event times.
- Conditional intensity allows us to easily define distributions with properties such as global trend or burstiness. -->

### Why do we need TPPs?
The most obvious use-case for such models is prediction and forecasting ("When will the next event occur?", "How many orders will we receive next week?"), but other applications are possible, such as anomaly detection ("Does activity of this user deviate from her normal behavior?") and understanding the data ("Do events of type A influence events of type B?").

In this blog post, I will consider a simple scenario (but I will cover the case ... in one of the future posts).

Each event corresponds to an arrival time $t_i$.

Temporal point process (TPP) defines a distribution of such event sequences over some interval $$[0, T]$$.

A realization of a TPP consists of a sequence of increasing arrival times $$0 < t_1 < \dots < t_N < T$$. [We usually assume that our TPPs are *simple*. This means that (1) the number of events $N$ is finite almost surely (=with probability one) and (2) the arrival times $t_i$ are distinct, i.e. $t_i \ne t_j$ for all $i\ne j$.]

The key property of TPPs is that both the number of events $N$ as well their arrival times $t_i$ are random.

we can represent ... simply as a sequence of arrival times. 

A natural way to define a distribution over event sequences $(t_1, t_2, t_3, ...)$ is by specifying the conditional distributions $p_1(t_1), p_2(t_2 | t_1), p_3(t_3 | t_1, t_2), \dots$. 

Given 

```python
t = 0
arrival_times = []
while t < T:
	t ~ p_i(t_i | t_1, ..., t_{i-1})
	if t < T:
		arrival_times.append(t)
```

[Remember, 

It's important to remember that depending on the time of the first event $t_1$, we get different conditional distributions $p_2(t_2|t_1)$.

Fig: Some TPP realization on interval [0, T].

There exist two ways to look at this problem - machine learning view (autoregressive probabilistic models) and 

### TPP as an autoregressive model

- PDF, CDF and survival function
- From conditional PDF to the conditional hazard function
- Stitching together the hazard functions to obtain the conditional intensity
- Intensity simply as an alternative to the conditional PDFs

Difference between intensity and PDF:

- PDF: I am not at t_{i-1}, what is the probability that the next event will happen in [t + dt)?
- Survival: I have already got up to t and the event hasn't happened yet. What is the probability that I will see it in [t + dt)?

### TPP as a counting process

There is another view

- Intensity as expectation of the counting process
- Conditional intensity for more general dependencies

Both views are, in fact, equivalent

### Why intensity function?

- Easy to specify certain types of dependencies (global trend, self-exciting, self-correcting), especially when we have marks
- Leads to fast inference and sampling - see Parts 2 & 4
- Not always they best way to look at things - more in Part 3

Intensity allows to define distributions in a piecewise manner - external event happens and changes 

### Examples of TPP models

- Inhomogeneous Poisson - global trend
- Hawkes process - self-exciting
- Self-correcting process

In the next blog post, I will discuss the many ...  -->

# References

- ICML 2018 tutorial by Manuel Gomez Rodriguez and Isabel Valera [http://learning.mpi-sws.org/tpp-icml18/](http://learning.mpi-sws.org/tpp-icml18/)
- Lecture notes by Jakob Rasmussen [https://arxiv.org/abs/1806.00221](https://arxiv.org/abs/1806.00221)
- A tutorial by Marian-Andrei Rizoiu et al. [https://arxiv.org/abs/1708.06401](https://arxiv.org/abs/1708.06401)
- A tutorial on Hawkes processes by Caner Turkmen [https://hawkeslib.readthedocs.io/en/latest/tutorial.html](https://hawkeslib.readthedocs.io/en/latest/tutorial.html)


Text and some other stuff [^1]
and a bit more


[^1]: Hello world

<details>
<summary>This list is hidden</summary>
What happens here?
</details>


