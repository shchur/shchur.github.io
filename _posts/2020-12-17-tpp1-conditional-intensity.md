---
layout: distill
title: "Temporal Point Processes 1: The Conditional Intensity Function"
description: "How can we define generative models for variable-length event sequences in continuous time?"
date: 2020-12-17
comments: true
og_image: /assets/img/posts/tpp1/twitter_card.jpg
twitter_title: "Temporal Point Processes 1: The Conditional Intensity Function"

authors:
  - name: Oleksandr Shchur

redirect_from:
  - /2020/12/17/tpp1-conditional-intensity.html

---

<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div><a href="#what-is-a-point-process"> What is a point process?</a></div>
    <div><a href="#tpp-as-an-autoregressive-model">TPP as an autoregressive model</a></div>
    <div><a href="#defining-tpps-using-the-conditional-intensity-function">Defining TPPs using the conditional intensity function</a></div>
    <div><a href="#tpp-as-a-counting-process">TPP as a counting process</a></div>
    <div><a href="#summary">Summary</a></div>
  </nav>
</d-contents>

## TL;DR
- Temporal point processes (TPPs) are probability distributions over variable-length event sequences in continuous time.
- We can view a TPP as an autoregressive model or as a counting process.
- The conditional intensity function $\lambda^*(t)$ connects these two viewpoints and allows us to specify TPPs with different behaviors, such as a global trend or burstiness.
- The conditional intensity $\lambda^*(t)$ is one of many ways to define a TPP &#8212; as an alternative, we could, for example, specify the conditional PDFs of the arrival times $$\{f_1^*, f_2^*, f_3^*, ...\}$$.

## What is a point process?
Probabilistic generative models are the bread and butter of modern machine learning.
They allow us to make predictions, find anomalies and learn useful representations of the data.
Most of the time, applying the generative model involves learning the probability distribution $$P(\boldsymbol{x})$$ over our data points $$\boldsymbol{x}$$.

We know what to do if $$\boldsymbol{x}$$ is a vector in $$\mathbb{R}^D$$
--- simply use a multivariate Gaussian or, if we need something more flexible, our favorite [normalizing flow](https://arxiv.org/abs/1912.02762) model.
But what if a single realization of our probabilistic model corresponds to a *set* of vectors $$\{\boldsymbol{x}_1, ..., \boldsymbol{x}_N\}$$?
Even worse, what if both $$N$$, the number of the vectors, as well as their locations $$\boldsymbol{x}_i$$ are random?
This is not some hypothetical scenario --- processes generating such data are abundant in the real world: 
<ul style="margin-top: 0px">
  <li> Transactions generated each day in a financial system</li>
  <li> Locations of disease outbreaks in a city, recorded each week</li>
  <li> Times and locations of earthquakes in some geographic region within a year</li>
</ul>

Point processes provide a framework for modeling and analyzing such data.
Each realization of a point process is a set $$\{\boldsymbol{x}_1, \dots, \boldsymbol{x}_N\}$$ consisting of a random number $$N$$ of *points* $$\boldsymbol{x}_i$$ that live in some space $$\mathcal{X}$$, hence the name "point process".
Depending on the choice of the space $$\mathcal{X}$$, we distinguish among different types of point processes.
For example, $$\mathcal{X} \subseteq \mathbb{R}^D$$ corresponds to a so-called *spatial point process*, where every point $$\boldsymbol{x}_i$$ can be viewed as a random location in space (e.g., a location of a disease outbreak).


<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp1/spp_sample.png" 
style="display: block; width: 90%; margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> 
Two realizations of a spatial point process on $\mathbb{R}^2$.
</figcaption>
</div>


Another important case, to which I will dedicate the rest of this post (and, hopefully, several future ones), are *temporal point processes* (TPPs), defined on the real half-line $$\mathcal{X} \subseteq [0, \infty)$$.
We can interpret the points in a TPP as events happening in continuous time, and therefore usually denote them as $$t_i$$ (instead of $$\boldsymbol{x}_i$$).

<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp1/tpp_sample.png" 
style="margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> 
Two realizations of a temporal point process on $[0, T]$.
</figcaption>
</div>

At first it might seem like TPPs are just a (boring) special case of spatial point processes, but this is not true.
Because of the ordered structure of the set $$[0, \infty)$$, we can treat TPP realizations (i.e., sets $$\{t_1, \dots, t_N\}$$) as ordered sequences $$\boldsymbol{t} = (t_1, \dots, t_N)$$, where $$t_1 < t_2 < \dots < t_N$$.
Additionally, we typically assume that the arrival time of the event $$t_i$$ is only influenced by the events that happened in the past.
As we will see in the next section, this makes specifying TPP distributions rather easy.
In contrast, spatial point processes don't permit such ordering on the events, and because of this often have intractable densities.


The theory of temporal point processes was mostly developed near the middle of the 20th century, taking roots in measure theory and stochastic processes.
For this reason, the notation and jargon used in TPP literature may sound strange and unfamiliar to people with a machine learning background (at least it did to me back when I started learning about TPPs).
In reality, though, most TPP-related concepts can be easily translated into the familiar language of probabilistic machine learning.

In this post we will investigate different ways to represent a TPP.
As we will see, a TPP can be treated as an autoregressive model or as a counting process.
We will learn about the conditional intensity function $$\lambda^*(t)$$ --- a central concept in the theory of point processes --- that unites these two perspectives and allows us to compactly describe various TPP distributions.



## TPP as an autoregressive model
How do we define a probabilistic model that generates variable-length event sequences $$\boldsymbol{t} = (t_1, \dots, t_N)$$<d-footnote>Some technicalities: We usually assume that our TPPs are <i>simple</i>. This means that (1) the number of events $N$ is finite almost surely (=with probability one) and (2) the arrival times $t_i$ are distinct, i.e. $t_i \ne t_j$ for all $i\ne j$. Additionally, we assume that $t_i$'s are continuous random variables. This means, among other things, that $\Pr(t_i \in [a, b]) = \Pr(t_i \in (a, b))$, i.e., we shouldn't worry about the interval boundaries too much.</d-footnote> in the interval $$[0, T]$$? 
Thanks to the inherent ordering on the events, we could define our model autoregressively.
We start by sampling $$t_1$$, the time of the first event, from some probability distribution $$P_1(t_1)$$ that is supported on $$[0, \infty)$$.
If $$t_1 > T$$, i.e., the event happened outside of the observed interval, we are done --- our realization $$\boldsymbol{t}$$ is just an empty sequence.
Otherwise, we sample the next event $$t_2$$ from the conditional distribution $$P_2(t_2 | t_1)$$ that is supported on $$[t_1, \infty)$$.
Again, we check if $$t_2 > T$$, and if not, proceed to sample $$t_3$$ from $$P_3(t_3 | t_1, t_2)$$.
We repeat this process until some event $$t_{N+1}$$ falls outside of the observed interval, at which point we stop the process and get our sample consisting of $$N$$ events.

At each step we are dealing with the conditional distribution of the event $$t_i$$ given the *history* of the past events $$\mathcal{H}_{t_i} = \{t_j: t_j < t_i\}$$.
We usually denote this distribution as $$P_i(t_i | \mathcal{H}_{t_i})$$. 
In the literature, you will also often meet the shorthand notation $$P_i^*(t_i)$$, where the star reminds us of the dependency on past events.
The important question is how to represent the probability distribution $$P_i^*(t_i)$$.


In machine learning, we usually characterize a continuous probability distribution $$P_i^*$$ by specifying its probability density functions (PDF) $$f_i^*$$.
Loosely speaking, the value $$f_i^*(t) dt$$ represents the probability that the event $$t_i$$ will happen in the interval $$[t, t + dt)$$, where $$dt$$ is some infinitesimal positive number.<d-footnote>All the explanations involving $dt$ are <a href="https://en.wikipedia.org/wiki/Differential_(infinitesimal)">not 100% rigorous</a> and are used to provide intuition — if we set $dt$ to some tiny positive number, then the equations would be approximately correct. Turning such a handwavy explanation into a rigorous mathematical argument would require taking the limit $dt \to 0$.</d-footnote>

However, there exist other ways to describe a distribution that might be more useful in certain contexts.
For example, the *cumulative distribution function* (CDF) $$F_i^*(t) = \int_0^{t} f_i^*(u) du$$ tells us the probability that the event $$t_i$$ will happen before time $$t$$.
Closely related is the *survival function* (SF), defined as $$S_i^*(t) = 1 - F_i^*(t)$$, which tells us the probability that the event $$t_i$$ will happen *after* time $$t$$. 

<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp1/pdf_cdf_sf.png" 
style="display: block; width: max(80%, 200px); margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> 
Interpretation of the PDF, CDF and SF. Here $\mathcal{H}_t = \{t_1, ..., t_{i-1}\}$.
</figcaption>
</div>

Finally, a lesser known option is the [*hazard function*](https://en.wikipedia.org/wiki/Failure_rate) $$h_i^*$$ that can be computed as $$h^*_i(t) = f_i^*(t) / S_i^*(t)$$.
The value $$h_i^*(t)dt$$ answers the question "What is the probability that the event $$t_i$$ will happen in the interval $$[t, t + dt)$$ given that it didn't happen before $$t$$?".
Let's look at this definition more closely to examine the connection between the PDF $$f_i^*$$ and the hazard function $$h_i^*$$.

Consider the following scenario.
The most recent event $$t_{i-1}$$ has just happened and our clock is at time $$t_{i-1}$$.
The value $$f_i^*(t)dt$$ tells us the probability that the next event $$t_i$$ will happen in $$[t, t+ dt)$$ (see next figure --- top).
Then, some time has elapsed, our clock is now at time $$t$$ and the event $$t_{i}$$ hasn't yet happened.
At this point in time, $$f_i^*(t)dt$$ is not equal to $$\Pr(t_i \in [t, t + dt) | \mathcal{H}_t)$$ anymore --- we need to condition on the fact that $$t_i$$ didn't happen before $$t$$.
For this, we renormalize the PDF such that it integrates to $$1$$ over the interval $$[t, \infty)$$ (see next figure --- center).

$$f_i^*(t | t_i \ge t) = \frac{f_i^*(t)}{\int_t^\infty f_i^*(u) du} =\frac{f_i^*(t)}{S_i^*(t)} =: h_i^*(t)$$

This value of the renormalized PDF exactly corresponds to the hazard function $$h_i^*$$ at time $$t$$ (see next figure --- bottom).

<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp1/renorm_pdf.png" 
style="display: block; width: max(80%, 200px); margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> 
PDF of $t_i$ (top), PDF of $t_i$ conditioned on $t_i \ge t$ (center), hazard function of $t_i$ (bottom).
</figcaption>
</div>

We can also go in the other direction and compute the PDF $$f_i^*$$ using $$h_i^*$$.
First, we need to compute the survival function

$$
\begin{aligned}
h_i^*(t) &= \frac{f_i^*(t)}{S_i^*(t)} = \frac{- \frac{d}{dt} S_i^*(t)}{S_i^*(t)} = -\frac{d}{dt} \log S_i^*(t)\\
& \Leftrightarrow S_i^*(t) = \exp \left( -\int_{t_{i-1}}^t h_i^*(u) du \right)
\end{aligned}
$$

This, in turn, allows us to obtain the PDF as

$$
\begin{aligned}
f_i^*(t) &= -\frac{d}{dt}S_i^*(t)\\
& = -\frac{d}{dt}\exp \left( -\int_{t_{i-1}}^t h_i^*(u) du \right)\\
&= h_i^*(t) \exp \left( -\int_{t_{i-1}}^t h_i^*(u) du \right)
\end{aligned}
$$

The name "hazard function" comes from the field of [survival analysis](https://en.wikipedia.org/wiki/Survival_analysis), where the goal is to predict hazardous events such as death of a patient or failure of some system.
In such a setting, the hazard function $$h_i^*$$ is often considered to be more interpretable<d-footnote>In my opinion, only very basic hazard functions are somewhat interpretable --- for example, if the hazard function monotonically increases as $t \to \infty$ (e.g., older people are more likely to die at any given time) or decreases (e.g., if a light bulb didn't break in the first hour of operation, then it's not defect and will serve for a long time). However, if your hazard function is defined by a neural network (e.g., a normalizing flow), I would argue that it's as (un)interpretable as the PDF $f_i^*.$</d-footnote> than the PDF $f_i^*$.
For example, if a system hasn't failed by time $$t_i$$, the value $$h_i^*(t)dt$$ corresponds to the probability of failure in the immediate future.
This quantity can be of interest when planning treatments or allocating resources.


Let's get back to our problem of characterizing the conditional distributions of a TPP.
We could specify any of the functions $$f_i^*$$, $$F_i^*$$, $$S_i^*$$ or $$h_i^*$$ (subject to the respective constraints<d-footnote>Constraints on PDF, CDF, SF and hazard function are necessary to ensure that they define a valid probability distribution. For example, a PDF $f_i^*$ must satisfy $f_i^*(t) \ge 0$ for all $t$ and $\int_{t_{i-1}}^\infty f_i^*(u) du = 1$. Similarly, a valid hazard function $h_i^*$ must satisfy $h_i^*(t) \ge 0$ for all $t$ and $\int_{t}^{\infty} h_i^*(u) du = \infty$ for all $t$.</d-footnote>), and each one of them would completely describe the distribution $$P_i^*$$.
Put differently, given one of these functions, we can directly compute the other three.
It's worth noting that there exist other functions (besides the four mentioned above) that we could use to describe a distribution.

<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp1/pdf_cdf_sf_hazard.png" />
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> 
Four ways to represent the conditional distribution $P_i^*(t)$: probability density function $f_i^*$, cumulative distribution function $F_i^*$, survival function $S_i^*$, and hazard function $h_i^*$.
</figcaption>
</div>

In summary, to define the full distribution of some TPP, we could, for instance, specify the conditional PDFs $$\{f_1^*, f_2^*, f_3^*, \dots\}$$
or, equivalently, the conditional hazard functions $$\{h_1^*, h_2^*, h_3^*, \dots\}$$.
However, dealing with all the different conditional distributions and their indices can be unwieldy.
Instead, we could consider yet another way of characterizing the TPP --- using the *conditional intensity function*.
The conditional intensity, denoted as $$\lambda^*(t)$$, is defined by stitching together the conditional hazard functions:

$$
\lambda^*(t) =
\begin{cases}
h_1^*(t) & \text{ if } 0 \le t \le t_1 \\
h_2^*(t) & \text{ if } t_1 < t \le t_2 \\
& \vdots\\
h_{N+1}^*(t) & \text{ if } t_N < t \le T \\
\end{cases}
$$

which can graphically be represented as follows:

<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp1/stitching.png" 
style="display: block; width: max(80%, 200px); margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> 
The conditional intensity $\lambda^*(t)$ is obtained by stitching together the hazard functions $h_i^*(t)$.
</figcaption>
</div>

Let's take a step back and remember what the $$*$$ notation means here.
When we write $$\lambda^*(t)$$, we actually mean $$\lambda(t | \mathcal{H}_t)$$.
That is, the conditional intensity function takes as input two arguments: (1) the current time $$t$$ and (2) the set of the preceding events $$\mathcal{H}_t$$ that can be of arbitrary size.
<!-- Similarly, the conditional hazard function $$h_i^*$$ also always depends on the past events. -->

We can turn the previous statement around:
To define a TPP distribution, we simply need to define some non-negative function<d-footnote>The conditional intensity must be non-negative since it's a ratio of two non-negative numbers — the PDF $f_i^*(t)$ and the SF $S_i^*(t)$. There is another technical detail: it should hold for any $t$ and $\mathcal{H}_t$ that $\int_t^{\infty} \lambda(u \vert \mathcal{H}_t) du = \infty$. If the latter condition is not fulfilled, the respective conditional PDF $f_i^*$ won't integrate to 1.</d-footnote> that takes as input the time $$t \in [0, T]$$ and a variable-sized set of past events $$\{t_1, \dots, t_{i-1}\}$$.
This will completely specify the conditional intensity $$\lambda^*(t)$$.
Given $$\lambda^*(t)$$, we can easily recover the conditional hazard functions $$h_i^*$$.
Finally, we can obtain the conditional PDFs $$f_i^*$$ from the $$h_i^*$$'s. 
Thus, we have completely specified our TPP distribution. Neat!

## Defining TPPs using the conditional intensity function

The main advantage of the conditional intensity is that it allows to compactly represent various TPPs with different behaviors.
For example, we could define a TPP where the intensity is independent of the history and only depends on the time $$t$$.

$$
\lambda^*(t) = g(t)
$$

This corresponds to the famous [Poisson process](https://en.wikipedia.org/wiki/Poisson_point_process).
High values of $$g(t)$$ correspond to a higher rate of event occurrence, so the Poisson process allows us to capture global trends.
For instance, we could use it to model passenger traffic in a subway network within a day.
More events (i.e., ticket purchases) happen in the morning and in the evening compared to the middle of the day, which is reflected by the variations in the intensity $$g(t)$$.


<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp1/poisson.png" 
style="display: block; width: max(80%, 200px); margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> 
A realization of a Poisson process (bottom) and the respective intensity function (top).
</figcaption>
</div>
The Poisson process has a number of other interesting properties and probably deserves a blog post of its own.


Another popular example is the [self-exciting process (a.k.a. Hawkes process)](https://arxiv.org/abs/1708.06401) with the conditional intensity function

$$
\lambda^*(t) = \mu + \sum_{t_j \in  \mathcal{H}_t} \alpha \exp(-(t - t_j))
$$

As we see above, the intensity increases by $$\alpha$$ whenever an event occurs and then exponentially decays towards the baseline level $$\mu$$.
Such an intensity function allows us to capture "bursty" event occurrences --- events often happen in quick succession.
For example, if a neuron fires in the brain, it's likely that this neuron will fire again in the near future.


<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp1/hawkes.png" 
style="display: block; width: max(80%, 200px); margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> 
A realization of a Hawkes process (bottom) and the respective intensity function (top).
</figcaption>
</div>

Both of the above examples could equivalently be specified using the conditional PDFs $$f_i^*$$ or the hazard functions $$h_i^*$$.
However, their description in terms of the conditional intensity $$\lambda^*(t)$$ is more elegant and compact --- we don't have to worry about the indices $$i$$, and we can understand the properties of respective TPPs (such as global trend or burstiness) by simply looking at the definition of $$\lambda^*(t)$$.



## TPP as a counting process

So far, we have represented TPP realizations as variable-length sequences, but this is not the only possible option. 
In many textbooks and (especially older) papers a TPP is defined as a *counting process* <d-footnote>A counting process is a special type of stochastic processes. A <a href="https://en.wikipedia.org/wiki/Stochastic_process">stochastic process</a> is a collection of random variables that are indexed by a real number. In our case, any real number $t \in [0, T]$ corresponds to a random variable $N(t).$</d-footnote> --- a probability distribution over functions.
Each realization of a counting process is an increasing function $$N \colon [0, T] \to \mathbb{N}_0$$.
We can think of $$N(t)$$ as the number of events that happened before time $$t \in [0, T]$$.

It's easy to see that this formulation is equivalent to the one we used before.
We can represent an event sequence $$\boldsymbol{t} = (t_1, \dots, t_N)$$ as a realization of a counting process by defining

$$
N(t) = \sum_{i=1}^N \mathbb{I}(t_i \le t)
$$

where $$\mathbb{I}$$ is the indicator function.


<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp1/counting.png"
style="display: block; width: max(80%, 200px); margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> 
Realization of a counting process (above) and the respective event sequence (below).
</figcaption>
</div>
Last, we will consider is how to characterize the distribution of a counting process.
Not surprisingly, the conditional intensity function $$\lambda^*(t)$$ that we defined in the previous section will again come up here.

Like before, suppose that $$dt$$ is an infinitesimal positive number.
We will consider the expected change in $$N(t)$$ during $$dt$$ given the history of past events $$\mathcal{H}_t$$, that is<d-footnote>The first equality follows from the assumption that $dt$ is small enough that at most a single event can happen in the interval $[t, t+ dt)$. Therefore, $N(t + dt) - N(t)$ can take only two values: either 1 or 0. The outcome "$N(t + dt) - N(t) = 1$" can be rephrased as "the event $t_i$ happened in the interval $[t, t+ dt)$ given that it didn't happen in the interval $[t_{i-1}, t)$" (where $t_{i-1}$, without loss of generality, is the last event that happened before time $t$). The probability of this can be computed using the conditional hazard function $h_i^*(t)$, which by definition is equal to the conditional intensity $\lambda^*(t)$.</d-footnote>

$$
\begin{aligned}
\mathbb{E}[N(t + dt) - N(t) | \mathcal{H}_t] =& \;1 \cdot \Pr(\text{next event } t_i \text{ happens in } [t, t + dt) | \mathcal{H}_t)\\
&+ 0 \cdot \Pr(\text{no event in } [t, t + dt) | \mathcal{H}_t)\\
=& \Pr(t_i \in [t, t+ dt) | \mathcal{H}_{t})\\
=& h_i^*(t) dt\\
=& \lambda^*(t) dt\\
\end{aligned}
$$


By rearranging the above equation we could define the conditional intensity function as

$$
\lambda^*(t) = \lim_{dt \to 0} \frac{\mathbb{E}[N(t + dt) - N(t) | \mathcal{H}_t]}{dt}
$$

which means, in simple words, that the conditional intensity is the expected number of events in a TPP per unit of time.

## Summary
We have uncovered the mystery of the name "temporal point process":

- **Process** --- a TPP can be defined as a counting *process*
- **Point** --- we can view each TPP realization $\boldsymbol{t} = (t_1, \dots, t_N)$ as a set of *"points"*
- **Temporal** --- we can interpret the "points" $t_i$ as arrival *times* of events

We learned about different ways to specify a TPP, such as using the conditional intensity $$\lambda^*(t)$$ or the conditional PDFs $$\{f_1^*, f_2^*, f_3^*, \dots\}$$.

In the next post of this series, I will talk about how we can put this theory to practice and implement neural-network-based TPP models.


### Acknowledgments
I would like to thank [Johannes Klicpera](https://twitter.com/klicperajo) for his feedback on this post.


### Further reading

<ul >
  <li> <a href="http://learning.mpi-sws.org/tpp-icml18/">ICML 2018 tutorial by Manuel Gomez Rodriguez and Isabel Valera</a></li>
  <li> <a href="https://arxiv.org/abs/1806.00221">Lecture notes by Jakob Rasmussen</a></li>
  <li> <a href="https://arxiv.org/abs/1708.06401">A tutorial by Marian-Andrei Rizoiu et al.</a></li>
  <li> <a href="https://hawkeslib.readthedocs.io/en/latest/tutorial.html">A tutorial on Hawkes processes by Caner Turkmen</a></li>
</ul>