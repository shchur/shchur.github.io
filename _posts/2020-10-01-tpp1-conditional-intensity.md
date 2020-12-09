---
layout: post
comments: true
title: "Temporal Point Processes 1: The Conditional Intensity Function"
subtitle: "How can we define generative models for variable-length event sequences in continuous time?"
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


<img src="/img/posts/tpp1/spp_sample.png" width="100%">
*Figure: Two realizations of a spatial point process on $$\mathbb{R}^2$$.*
{: style="text-align: center;"}


Another important case, to which I will dedicate the rest of this post (and, hopefully, several future ones), are *temporal point processes* (TPPs), defined on the real half-line $$\mathcal{X} \subseteq [0, \infty)$$.
We can interpret the points in a TPP as events happening in continuous time, and therefore usually denote them as $$t_i$$ (instead of $$\bm{x}_i$$).

<img src="/img/posts/tpp1/tpp_sample.png" width="100%">
*Figure: Two realizations of a temporal point process on $$[0, T]$$.*
{: style="text-align: center;"}

At first it might seem like TPPs are just a (boring) special case of spatial point processes, but this is not true.
Because of the ordered structure of the set $$[0, \infty)$$, we can treat TPP realizations (i.e., sets $$\{t_1, \dots, t_N\}$$) as ordered sequences $$\bm{t} = (t_1, \dots, t_N)$$, where $$t_1 < t_2 < \dots < t_N$$.
Additionally, we typically assume that the arrival time of the event $$t_i$$ is only influenced by the events that happened in the past.
As we will see in the next section, this makes specifying TPP distributions rather easy.
In contrast, spatial point processes don't permit such ordering on the events, and because of this often have intractable densities.


The theory of temporal point processes was mostly developed near the middle of the 20th century, taking roots in measure theory and stochastic processes.
For this reason, the notation and jargon used in TPP literature may sound strange and unfamiliar to people with a machine learning background (at least it did to me).
In reality, though, most TPP-related concepts can be easily translated into the familiar language of probabilistic machine learning.
In the rest of this post I will talk about these two viewpoints and show that they are equivalent.

The first question that we need to answer "How can we describe a TPP distribution?".
By far the most popular approach that you will find in virtually every TPP paper or textbook is to use the *conditional intensity function*, commonly denoted as $$\lambda^*(t)$$.


The fundamental concept that you will find in virtually every TPP paper or textbook is the *conditional intensity function*, commonly denoted as $$\lambda^*(t)$$.
We will now look at it from two angles (autoregressive view vs. counting process view) and see how the conditional intensity allows us to specify TPP distributions.


### TPP as an autoregressive model
How do we define a probabilistic model that generates variable-length event sequences $$\bm{t} = (t_1, \dots, t_N)$$[^1] in the interval $$[0, T]$$? 
Thanks to the inherent ordering on the events, we could define our model autoregressively.
We start by sampling $$t_1$$, the time of the first event, from some probability distribution $$p_1(t_1)$$ that is supported on $$[0, \infty)$$.
If $$t_1 > T$$, i.e., the event happened outside of the observed interval, we are done --- our realization $$\bm{t}$$ is just an empty sequence.
Otherwise, we sample the next event $$t_2$$ from the conditional distribution $$p_2(t_2 | t_1)$$ that is supported on $$[t_1, \infty)$$.
Again, we check if $$t_2 > T$$, and if not, proceed to sample $$t_3$$ from $$p_3(t_3 | t_1, t_2)$$.
We keep repeating this process until some event $$t_{N+1}$$ falls outside of the observed interval, at which point we stop the process and get our sample consisting of $$N$$ events.

At each step we are dealing with the conditional distribution of the event $$t_i$$ given the *history* of the past events $$\mathcal{H}_{t_i} = \{t_j: t_j < t_i\}$$.
We usually denote this distribution as $$p_i(t_i | \mathcal{H}_{t_i})$$. 
In the literature, you can also often meet the shorthand notation $$p_i^*(t_i)$$, where the star reminds us of the dependency on the past events.
The important question is how to represent the probability distribution $$p_i^*(t_i)$$.


In machine learning, we usually characterize a continuous probability distribution $$p_i^*$$ by specifying its probability density functions (PDF) $$f_i^*$$.
Loosely speaking, the value $$f_i^*(t) dt$$ represents the probability that the event $$t_i$$ will happen in the interval $$[t, t + dt)$$, where $$dt$$ is some infinitesimal positive number.

However, there exist other ways to describe a distribution that might be more useful in certain contexts.
For example, the *cumulative distribution function* (CDF) $$F_i^*(t) = \int_0^{t} f_i^*(u) du$$ tells us the probability that the event $$t_i$$ will happen before time $$t$$.
Closely related is the *survival function* (SF), defined as $$S_i^*(t) = 1 - F_i^*(t)$$, which tells us the probability the event $$t_i$$ will happen *after* time $$t$$. 

<img src="/img/posts/tpp1/pdf_cdf_sf.png" width="100%">
*Figure: Interpretation of the PDF, CDF and SF.*
{: style="text-align: center;"}


Finally, a lesser known option is the *hazard function* $$h_i^*$$ that can be computed as $$h^*_i(t) = f_i^*(t) / S_i^*(t)$$.
The value $$h_i^*(t)dt$$ answers the question "What is the probability that the event $$t_i$$ will happen in the interval $$[t, t + dt)$$ given that it didn't happen before $$t$$?".
Let's look at this definition more closely to examine the connection between the PDF $$f_i^*$$ and the hazard function $$h_i^*$$.

Consider the following scenario.
The most recent event $$t_{i-1}$$ has just happened and our clock is at time $$t_{i-1}$$.
The value $$f_i^*(t)dt$$ tells us the probability that the next event $$t_i$$ will happen in $$[t, t+ dt)$$.
Then, some time has elapsed, our clock is now at time $$t$$ and the event $$t_{i}$$ hasn't yet happened.
At this point in time, $$f_i^*(t)dt$$ is not equal to $$\Pr(t_i \in [t, t + dt) | \mathcal{H}_t)$$ anymore --- we need to condition on the fact that $$t_i$$ didn't happen before $$t$$.
For this, we renormalize the PDF such that it integrates to $$1$$ over the interval $$[t, \infty)$$.

$$f_i^*(t | t_i > t) = \frac{f_i^*(t)}{\int_t^\infty f_i^*(u) du} =\frac{f_i^*(t)}{S_i^*(t)} =: h_i^*(t)$$

This value of the renormalized PDF exactly corresponds to the hazard function $$h_i^*$$ at time $$t$$.
The name "hazard function" comes from the field of [survival analysis](https://web.stanford.edu/~lutian/coursepdf/unit1.pdf), where the goal is to predict hazardous events such as death of a patient or failure of some system.
In such a setting, the hazard function $$h_i^*$$ is often considered to be more interpretable[^2] than the PDF $$f_i^*$$.
For example, if a system hasn't failed by time $$t_i$$, the quantity $$h_i^*(t)dt$$ corresponds to the probability of failure in the immediate future, which can be useful when planning treatments or allocating resources.

<img src="/img/posts/tpp1/pdf_cdf_sf_hazard.png" width="100%">
*Figure: Four ways to represent the conditional distribution $$p_i^*(t)$$.*
{: style="text-align: center;"}

Let's get back to our problem of characterizing the conditional distributions of a TPP.
We could specify any of the functions $$f_i^*$$, $$F_i^*$$, $$S_i^*$$ or $$h_i^*$$ (subject to the respective constraints), and each one of them would completely describe the distribution $$p_i^*$$.
Put differently, given one of these functions, we can easily compute the other three.

<details>
<summary>Computing the different functions from each other</summary>
</details>

<!-- In point process literature, we usually consider the hazard function $$h_i^*$$.
This happens for traditional reasons, but also because hazard functions are often more interpretable and easier to describe when talking about popular simple models. -->

This means, to define the full distribution of some TPP, we could, for instance, specify the conditional PDFs $$\{f_1^*, f_2^*, f_3^*, \dots\}$$
or, equivalently, the conditional hazard functions $$\{h_1^*, h_2^*, h_3^*, \dots\}$$[^2].
However, dealing with all the different conditional distributions and their indices can be unwieldy.
Instead, we could consider yet another way of characterizing the TPP --- using the *conditional intensity function*.
The conditional intensity, denoted as $$\lambda^*(t)$$, is defined by simply stitching together the conditional hazard functions:

$$
\lambda^*(t) =
\begin{cases}
h_1^*(t) & \text{ if } 0 \le t \le t_1 \\
h_2^*(t) & \text{ if } t_1 < t \le t_2 \\
& \vdots\\
h_{N+1}^*(t) & \text{ if } t_N < t \le T \\
\end{cases}
$$

**Figure: Stitching hazard functions**

Let's take a step back and remember what the $$*$$ notation means here.
When we write $$\lambda^*(t)$$, we actually mean $$\lambda(t | \mathcal{H}_t)$$.
That is, the conditional intensity function takes as input two arguments: (1) the current time $$t$$ and (2) the set of the preceding events $$\mathcal{H}_t$$ that can be of arbitrary size.
<!-- Similarly, the conditional hazard function $$h_i^*$$ also always depends on the past events. -->

We can turn the previous statement around:
To define a TPP distribution, we simply need to define some non-negative[^2] function that takes as input the time $$t \in [0, T]$$ and a variable-sized set of past events $$\{t_1, \dots, t_{i-1}\}$$.
This will completely specify the conditional intensity $$\lambda^*(t)$$.
Given $$\lambda^*(t)$$, we can easily recover the conditional hazard functions $$h_i^*$$.
Finally, we can obtain the conditional PDFs $$f_i^*$$ from the $$h_i^*$$'s. 
Thus, we have completely specified our TPP distribution. Neat!

### Defining TPPs using the conditional intensity function

The main advantage of the conditional intensity is that it allows to compactly represent various TPPs with different behaviors.
For example, we could define a TPP where the intensity is independent of the history and only depends on the time $$t$$.

$$
\lambda(t | \mathcal{H}_t) = g(t)
$$

This corresponds to the famous [Poisson process](https://www.wikiwand.com/en/Poisson_point_process).
High values of $$g(t)$$ correspond to a higher rate of event occurrence, so we could use a Poisson process to model global trends.
The Poisson process has a number of other interesting properties and deserves a blog post of its own.

**Figure: Realizations from a Poisson process**



Another popular example is the self-exciting process (a.k.a. Hawkes process) with the conditional intensity function

$$
\lambda(t | \mathcal{H}_t) = \mu + \sum_{\substack{t_j \in \, \mathcal{H}_t\\t_j  < \, t \;\;}} \alpha \exp(-(t - t_j))
$$

As we see above, the intensity increases by $$\alpha$$ whenever an event occurs and then exponentially decays towards the baseline level $$\mu$$.
Such an intensity function allows us to capture "bursty" event occurrences --- events often happen in quick succession.


**Figure: Realizations from a Hawkes process**





### TPP as a counting process

So far, 


*counting process*.


**Figure: Different representations of an event sequence **

- Intensity as expectation of the counting process
- Conditional intensity for more general dependencies

Both views are, in fact, equivalent

### Summary
Temporal point processes define probability distributions over variable-length event sequences.
We can view TPPs as autoregressive models.
Alternatively, a TPP can be represented as a counting process $$\{N(t)\}_{t=0}^{T}$$.

The conditional intensity function $$\lambda^*(t)$$ connects these two viewpoints and allows us to specify temporal point processes with different behaviors.

# References

- ICML 2018 tutorial by Manuel Gomez Rodriguez and Isabel Valera [http://learning.mpi-sws.org/tpp-icml18/](http://learning.mpi-sws.org/tpp-icml18/)
- Lecture notes by Jakob Rasmussen [https://arxiv.org/abs/1806.00221](https://arxiv.org/abs/1806.00221)
- A tutorial by Marian-Andrei Rizoiu et al. [https://arxiv.org/abs/1708.06401](https://arxiv.org/abs/1708.06401)
- A tutorial on Hawkes processes by Caner Turkmen [https://hawkeslib.readthedocs.io/en/latest/tutorial.html](https://hawkeslib.readthedocs.io/en/latest/tutorial.html)



[^1]: Some technicalities: We usually assume that our TPPs are *simple*. This means that (1) the number of events $$N$$ is finite almost surely (=with probability one) and (2) the arrival times $$t_i$$ are distinct, i.e. $$t_i \ne t_j$$ for all $$i\ne j$$. Additionally, we assume that $$t_i$$'s are continuous random variables. This means, among other things, that $$\Pr(t_i \in [a, b]) = \Pr(t_i \in (a, b))$$, i.e., we shouldn't worry about the interval boundaries too much.

[^2]: Another footnote.
