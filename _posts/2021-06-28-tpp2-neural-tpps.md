---
layout: distill
title: "Temporal Point Processes 2: Neural TPP Models"
description: "How can we define flexible TPP models using neural networks?"
date: 2021-06-28
comments: true
og_image: /assets/img/posts/tpp2/twitter_card.jpg
twitter_title: "Temporal Point Processes 2: Neural TPP Models"

bibliography: 2021-06-28-tpp2.bib

authors:
  - name: Oleksandr Shchur


---

<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div><a href="#representing-the-data">Representing the data</a></div>
    <div><a href="#constructing-a-neural-tpp-model">Constructing a neural TPP model</a></div>
    <ul>
      <li><a href="#encoding-the-history-into-a-vector">Encoding the history into a vector</a></li>
      <li><a href="#choosing-a-parametric-distribution">Choosing a parametric distribution</a></li>
      <li><a href="#conditional-distribution">Conditional distribution</a></li>
    </ul>
    <div><a href="#likelihood-function">Likelihood function</a></div>
    <div><a href="#putting-everything-together">Putting everything together</a></div>
    <div><a href="#concluding-remarks">Concluding remarks</a></div>
  </nav>
</d-contents>

In this post, we will learn about [neural temporal point processes](https://arxiv.org/abs/2104.03528) (TPPs) --- flexible generative models for variable-length event sequences in continuous time.
More specifically, we will
- Learn how to parametrize TPPs using neural networks;
- Derive the likelihood function for TPPs (that we will use as the training objective for our model);
- Implement a neural TPP model in PyTorch.

If you haven't read the [previous post in the series](https://shchur.github.io/2020/12/17/tpp1-conditional-intensity.html), I recommend checking it out to get familiar with the main concepts and notation.
Alternatively, click on the arrow below to see a short recap of the basics.

<details>
  <summary markdown='span'>Recap</summary>
  A temporal point process (TPP) is a probability distribution over variable-length event sequnces in a time interval $[0, T]$.
  We can represent a realization of a TPP as a sequence of <i>strcitly increasing</i> arrival times $\boldsymbol{t} = (t_1, ..., t_N)$, where $N$, the number of events, is itself a random variable.
  We can specify a TPP by defining $P_i^*(t_i)$, the conditional distribution of the next arrival time $t_i$ given past events $\{t_1, \dots, t_{i-1}\}$, for $i = 1, 2, 3, \dots$.
  The distribution $P_i^*(t_i)$ can be specified with either a <a target="_blank" href="https://en.wikipedia.org/wiki/Probability_density_function">probability density function</a> (PDF) $f_i^*$, a <a target="_blank" href="https://en.wikipedia.org/wiki/Survival_function">survival function</a> (SF) $S_i^*$, or a <a target="_blank" href="https://en.wikipedia.org/wiki/Failure_rate">hazard function</a> (HF) $h_i^*$.
</details>



## Representing the data

We will define our neural TPP as an autoregressive model.
To do this, at each step $i = 1, 2, 3, ...$ we need to specify the distribution $$P_i^*(t_i) := P_i(t_i \vert \mathcal{H}_{t_i})$$ of the next arrival time $$t_i$$ given the history of past events $$\mathcal{H}_{t_i} = \{t_1, \dots, t_{i-1}\}$$.
An equivalent but more convenient approach is to instead work with the *inter-event* times $(\tau_1, \dots, \tau_{N+1})$,<d-footnote>Note that we represent an event sequence $(t_1, \dots, t_N)$ with $N$ events using $N+1$ inter-event times. The last inter-event time $\tau_{N+1}$ corresponds to the time from the last event until $T$, the end of the observed time interval.</d-footnote> where we compute $\tau_i = t_i - t_{i-1}$ (assuming $t_0 = 0$ and $t_{N+1} = T$).

<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp2/inter_times.png"
style="display: block; width: 80%; margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
An event sequence can equivalently be represented by the arrival times $(t_1, \dots, t_N)$ or the inter-event times $(\tau_1, \dots, \tau_{N+1})$.
</figcaption>
</div>

First, let's load some data and covnert it into a format that can be processed by our model.
You can find the dataset and a [Jupyter notebook](https://colab.research.google.com/github/shchur/shchur.github.io/blob/gh-pages/assets/notebooks/tpp2/neural_tpp.ipynb) with all the code used in this blog post [here](https://github.com/shchur/shchur.github.io/blob/gh-pages/assets/notebooks/tpp2/).

<d-code language="python">
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

data = torch.load("toy_dataset.pkl")
# arrival_times_list is a list of variable-length lists
# arrival_times_list[j] is the list of arrival times of sequence j
arrival_times_list = data["arrival_times"]
# t_end = length of the observerd time interval [0, t_end]
t_end = data["t_end"]
seq_lengths = torch.tensor(
    [len(t) for t in arrival_times_list], dtype=torch.long
)  # (B,)

def get_inter_times(t, t_end):
    tau = np.diff(t, prepend=0.0, append=t_end)
    return torch.tensor(tau, dtype=torch.float32)

inter_times_list = [get_inter_times(t, t_end) for t in arrival_times_list]
inter_times = pad_sequence(inter_times_list, batch_first=True)  # (B, L)

</d-code>

<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp2/preprocess_times.png"
style="display: block; width: 90%; margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
We convert 3 variable-length event sequences of arrival times $\boldsymbol{t}^{(1)}$, $\boldsymbol{t}^{(2)}$, $\boldsymbol{t}^{(3)}$ into a tensor of padded inter-event times of shape <tt>(B, L)</tt>, where <tt>B</tt> - batch size, <tt>L</tt> - padded length.
</figcaption>
</div>

## Constructing a neural TPP model

How can we parametrize the conditional distribution $$P_i^*(\tau_i)$$ with a neural network?
A simple and elegant answer to this question was proposed in the seminal work by Du, Dai, Trivedi, Gomez-Rodriguez and Song <d-cite key="du2016recurrent"></d-cite>:


1. Encode the event history $$\mathcal{H}_{t_i} = \{t_1, \dots, t_{i-1}\}$$ into a *fixed-dimensional* context vector $$\boldsymbol{c}_i \in \mathbb{R}^C$$ using a neural network.
2. Pick a parametric probability density function $$f(\cdot \vert \boldsymbol{\theta})$$ that defines the distribution of a positive<d-footnote>We always assume that the arrival times are sorted, that is, $t_i < t_{i+1}$ for all $i$. Therefore, the inter-event times $\tau_i$ are strictly positive.</d-footnote> random variable (e.g., PDF of the [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution) or [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution)).
3. Use the context vector $$\boldsymbol{c}_i$$ to obtain the parameters $$\boldsymbol{\theta}_i$$. Plug in $$\boldsymbol{\theta}_i$$ into $$f(\cdot \vert \boldsymbol{\theta})$$ to obtain the PDF $$f(\tau_i \vert \boldsymbol{\theta}_i)$$ of the conditional distribution $$P_i^*(\tau_i)$$.

We will now look at each of these steps in more detail.

<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp2/architecture.png"
style="display: block; width: 90%; margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
Schematic representation of the neural TPP model that we will implement today.
</figcaption>
</div>

### Encoding the history into a vector
The original approach by Du et al. <d-cite key="du2016recurrent"></d-cite> uses a recurrent neural network (RNN) to encode the event history into a vector.
This works as follows.

<ol type="i">
<li>Each event $t_j$ is represented by a feature vector $\boldsymbol{y}_j$.
In our model, we will simply define $\boldsymbol{y}_j = (\tau_j, \log \tau_j)^T$.<d-footnote>Using the logarithm here allows the model to distinguish between very small inter-event times.</d-footnote> More sophisticated approaches, like positional encoding with trigonometric functions <d-cite key="zhang2020self"></d-cite>, are also possible.</li>

<li>We initialize $\boldsymbol{c}_1$ (for example, to a vector of all zeros). The vector $\boldsymbol{c}_1$ will work both as the initial state of the RNN, as well as to obtain the parameters of $P_1^*(\tau_1)$.</li>

<li>After each event $t_i$, we compute the next the context vector $\boldsymbol{c}_{i+1}$ (i.e., the next hidden state of the RNN) based on the previous state $\boldsymbol{c}_i$ and features $\boldsymbol{y}_i$
$$
\boldsymbol{c}_{i+1} = \operatorname{Update}(\boldsymbol{c}_i, \boldsymbol{y}_{i}).
$$
The specific RNN architecture is not very important here &#8212; <a target="_blank" href="https://pytorch.org/docs/stable/generated/torch.nn.RNN.html">vanilla RNN</a>,
<a target="_blank" href="https://pytorch.org/docs/stable/generated/torch.nn.GRU.html">GRU</a> or
<a target="_blank" href="https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html">LSTM</a> update functions can all be used here.
By processing the entire sequence $(t_1, \dots, t_N)$, we compute all the context vectors $(\boldsymbol{c}_1, \dots, \boldsymbol{c}_{N+1})$.
</li>
</ol>

<d-code language="python">
import torch.nn as nn
import torch.nn.functional as F

context_size = 32
rnn = nn.GRU(input_size=2, hidden_size=context_size, batch_first=True)

def get_context(inter_times):
    # inter_times: Padded inter-event times, shape (B, L)
    tau = inter_times.unsqueeze(-1)  # (B, L, 1)
    # Clamp tau to avoid computing log(0) for padding and getting NaNs
    log_tau = inter_times.clamp_min(1e-8).log().unsqueeze(-1)  # (B, L, 1)
    rnn_input = torch.cat([tau, log_tau], dim=-1)  # (B, L, 2)
    # The intial state is automatically set to zeros
    rnn_output = rnn(rnn_input)  # (B, L, C)
    # Shift by one such that context[:, i] will be used
    # to parametrize the distribution of inter_times[:, i]
    context = F.pad(rnn_output[:, :-1, :], (0, 0, 1, 0))  # (B, L, C)
    return context
</d-code>


<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp2/padding.png"
style="display: block; width: 90%; margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
We shift the <tt>rnn_output</tt> by one and add padding to obtain the <tt>context</tt> tensor that is properly aligned with the <tt>inter_times</tt>.
</figcaption>
</div>

### Choosing a parametric distribution

When picking a parametric distribution for inter-event times, we have to make sure that its probability density function (PDF) $f(\cdot \vert \boldsymbol{\theta})$ and survival function $S(\cdot \vert \boldsymbol{\theta})$ can be computed analytically --- we will need this later when computing the log-likelihood.
For some applications, it's also nice to able to sample from the distribution analytically.
I decided to use <a target="_blank" href="https://en.wikipedia.org/wiki/Weibull_distribution#Alternative_parameterizations">Weibull distribution</a> here as it satisfies all these properties.<d-footnote>Many other distributions over $[0, \infty)$ also satisfy these properties (e.g., exponential, log-normal, log-logistic, Gompertz distributions, or their mixtures), but some don't. For example, computing the survival function and sampling are not straightforward for the gamma distribution.</d-footnote>

<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp2/weibull.png"
style="display: block; width: 100%; margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
PDF of the Weibull distribution with different values of the parameters $b$ and $k$.
</figcaption>
</div>

The Weibull distribution has two strictly positive parameters $\boldsymbol{\theta} = (b, k)$. The probability density function is computed as
\\[
f(\tau \vert b, k) = b k \tau^{k-1} \exp(-b \tau^{k})
\\]
and the survival function is
\\[
S(\tau \vert b, k) = \exp(-b \tau^{k}).
\\]

We implement the Weibull distribution using an API that is similar to <tt><a href="https://pytorch.org/docs/stable/distributions.html">torch.distributions</a></tt>.
<d-code language="python">
class Weibull:
    def __init__(self, b, k, eps=1e-8):
        # b and k are strictly positive tensors of the same shape
        self.b = b
        self.k = k
        self.eps = eps

    def log_prob(self, x):
        """Logarithm of the probability density function log(f(x))."""
        # x must have the same shape as self.b and self.k
        x = x.clamp_min(self.eps)  # pow is unstable for inputs close to 0
        return (self.b.log() + self.k.log() + (self.k - 1) * x.log()
                + self.b.neg() * torch.pow(x, self.k))

    def log_survival(self, x):
        """Logarithm of the survival function log(S(x))."""
        x = x.clamp_min(self.eps)
        return self.b.neg() * torch.pow(x, self.k)
</d-code>

### Conditional distribution

To obtain the conditional PDF $f^*_i(\tau_i) := f(\tau_i | k_i, b_i)$ of the next inter-event time given the history, we compute the parameters $k_i$, $b_i$ using the most recent context embedding $\boldsymbol{c}_i$
\\[
\begin{aligned}
k_i = \sigma(\boldsymbol{v}^T_k \boldsymbol{c}_i + d_k) & \qquad & b_i = \sigma(\boldsymbol{v}^T_b \boldsymbol{c}_i + d_b).
\end{aligned}
\\]
Here, $\boldsymbol{v}_k \in \mathbb{R}^C, \boldsymbol{v}_b \in \mathbb{R}^C, d_k \in \mathbb{R}, d_b \in \mathbb{R}$ are learnable weights, and
$\sigma: \mathbb{R} \to (0, \infty)$ is a nonlinear function that ensures that the parameters are strictly positive (e.g., softplus).

<d-code language="python">
hypernet = nn.Linear(in_features=context_size, out_features=2)

def get_inter_time_distribution(context):
    # context has shape (B, L, C)
    raw_params = hypernet(context)  # (B, L, 2)
    b = F.softplus(raw_params[..., 0])  # (B, L)
    k = F.softplus(raw_params[..., 1])  # (B, L)
    return Weibull(b, k)

</d-code>
The weights $\\{\boldsymbol{v}_k, \boldsymbol{v}_b, d_k, d_b\\}$ together with the weights of the RNN are the learnable parameters of our neural TPP model.
The next question we have to answer is how to estimate these from data.

## Likelihood function
Log-likelihood (LL) is the default training objective for generative probabilistic models, and TPPs are no exception.
To derive the likelihood function for a TPP we will start with a simple example.

Suppose we have observed a single event with arrival time $$t_1$$ in the time interval $$[0, T]$$.
We can describe this outcome as "the first event happened in $[t_1, t_1 + dt)$ and the second event happened after $T$". <d-footnote>Here $dt$ denotes an infinitesimal positive number.</d-footnote>
The probability of this outcome is

$$
\begin{align}
\begin{split}
p(\{t_1\}) =& \Pr(\text{1st event in $[t_1, t_1 + dt)$})\\ & \times \Pr(\text{2nd event after $T$} \mid t_1)\\
=& f_1^*(t_1) dt \times S_2^*(T)
\end{split}
\end{align}
$$

The equality here follows simply from the definition of the PDF $$f_1^*$$ and the survival function $$S_2^*$$ (as discussed in the [previous post](https://shchur.github.io/blog/2020/tpp1-conditional-intensity/)).<d-footnote>We can make another interesting observation here. When computing the probability in Equation (1), we could also consider the event
$$
  \{\text{3rd event after $T$} \mid \text{1st event at $t_1$, 2nd event after $T$} \}.
$$
However, the probability of this event is equal to 1, since we already know that the second event happened after $T$ and, by definition, the third event happens after the second event.
The same holds for events like $\{\text{4th event after $T$} \mid ... \}$.
Therefore, even though a TPP realization may contain an arbitrary large number of events, when computing the the likelihood of a particular sequence $\boldsymbol{t} = (t_1, \dots, t_N)$, we only need to consider $N+1$ terms.</d-footnote>
Following the same reasoning, we compute the likelihood for an arbitrary sequence
$$\boldsymbol{t} = (t_1, t_2, \dots, t_N)$$ consisting of $$N$$ events as

$$
\begin{align}
p(\boldsymbol{t}) = (dt)^N \left(\prod_{i=1}^{N} f_i^*(t_i)\right) S_{N+1}^*(T).
\end{align}
$$

The $(dt)^N$ term is just a multiplicative constant with respect to the model parameters, so we ignore it during optimization. By taking the logarithm, we get the log-likelihood

$$
\begin{align}
\log p(\boldsymbol{t}) &=  \left(\sum_{i=1}^{N} \log f_i^*(t_i)\right) + \log S_{N+1}^*(T).
\end{align}
$$

Lastly, by slightly abusing the notation, we switch back to the inter-event times and obtain

$$
\begin{align}
\log p(\boldsymbol{t}) &= \left(\sum_{i=1}^{N} \log f_i^*(\tau_i)\right) + \log S_{N+1}^*(\tau_{N+1}).
\end{align}
$$

We will use this formulation of the log-likelihood to train our neural TPP model.

It's worth noting that Equation (4) is not the only way to express the log-likelihood of a TPP.
In the previous post, we talked about different functions characterizing a TPP, such as conditional hazard functions $$\{h_1^*, h_2^*, h_3^*...\}$$ and the conditional intensity function $$\lambda^*(t)$$.
Many papers and textbooks work with these functions instead.
Click on the arrow below for more details.

<details>
  <summary markdown='span'>Other ways to compute the log-likelihood of a TPP</summary>

We combine Equation (3) with the definition of the conditional hazard function $h_i^*(t) = f_i^*(t)/S_i^*(t)$ and obtain

$$
\begin{aligned}
\log p(\boldsymbol{t}) =& \left(\sum_{i=1}^{N} \log f_i^*(t_i)\right) + \log S_{N+1}^*(T)\\
=& \left(\sum_{i=1}^{N} \log h_i^*(t_i) + \log S_i^*(t_i)\right) + \log S_{N+1}^*(T)\\
=& \sum_{i=1}^{N} \log h_i^*(t_i) + \sum_{i=1}^{N+1} \log S_i^*(t_i),
\end{aligned}
$$

where we defined $t_{N+1}=T$. Last time, we derived the equality <d-footnote>
From the definition of the hazard function, it follows

$$
\begin{aligned}
h_i^*(t) &= \frac{f_i^*(t)}{S_i^*(t)} = \frac{- \frac{d}{dt} S_i^*(t)}{S_i^*(t)} = -\frac{d}{dt} \log S_i^*(t)\\
& \Leftrightarrow S_i^*(t) = \exp \left( -\int_{t_{i-1}}^t h_i^*(u) du \right)
\end{aligned}
$$</d-footnote>

$$S_i^*(t) = \exp \left(-\int_{t_{i-1}}^{t} h_i^*(u) du\right).$$

Plugging this into the expression for the log-likelihood, we get

$$
\begin{aligned}
\log p(\boldsymbol{t}) =& \sum_{i=1}^{N} \log h_i^*(t_i) - \sum_{i=1}^{N+1} \left(\int_{t_{i-1}}^{t_i} h_i^*(u) du\right)
\end{aligned}
$$

Finally, using the definition of the conditional intensity,<d-footnote>
The conditional intensity is defined piecewise by stitching together the hazard functions

$$
\lambda^*(t) =
\begin{cases}
h_1^*(t) & \text{ if } 0 \le t \le t_1 \\
h_2^*(t) & \text{ if } t_1 < t \le t_2 \\
& \vdots\\
h_{N+1}^*(t) & \text{ if } t_N < t \le T \\
\end{cases}
$$
</d-footnote> we rewrite the log-likelihood as

$$
\begin{aligned}
\log p(\boldsymbol{t}) =& \sum_{i=1}^{N} \log \lambda^*(t_i) - \sum_{i=1}^{N+1} \left(\int_{t_{i-1}}^{t_i} \lambda^*(u) du\right)\\
=& \sum_{i=1}^{N} \log \lambda^*(t_i) - \int_{0}^{T} \lambda^*(u) du.
\end{aligned}
$$

You will often see this expression for the LL in papers and textbooks.
As we have just showed, it is exactly equivalent to both Equations (3) and (4) that we derived before.
</details>

<br>

The trickiest part when implementing TPP models is vectorizing the operations on variable-length sequences (i.e., avoiding for-loops).
This is usually done with masking and operations like <tt>torch.gather</tt>.
For example, here is a vectorized implementation of the negative log-likelihood (NLL) for our neural TPP model.
<d-code language="python">
def nll_loss(inter_times, seq_lengths):
    # inter_times: Padded inter-event times, shape (B, L)
    # seq_lengths: Number of events in each sequence, shape (B,)
    context = get_context(inter_times)  # (B, L, C)
    inter_time_dist = get_inter_time_distribution(context)

    log_pdf = inter_time_dist.log_prob(inter_times)  # (B, L)
    # Construct a boolean mask that selects observed events
    arange = torch.arange(inter_times.shape[1], device=seq_lengths.device)
    mask = (arange[None, :] < seq_lengths[:, None]).float()  # (B, L)
    log_like = (log_pdf * mask).sum(-1)  # (B,)

    log_surv = inter_time_dist.log_survival(inter_times)  # (B, L)
    end_idx = seq_lengths.unsqueeze(-1)  # (B, 1)
    log_surv_last = torch.gather(log_surv, dim=-1, index=end_idx)  # (B, 1)
    log_like += log_surv_last.squeeze(-1)  # (B,)
    return -log_like

</d-code>

<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp2/nll_computation.png"
style="display: block; width: 90%; margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
Log-likelihood of a sequence $(\tau_1, ..., \tau_{N+1})$ is computed as $\left(\sum_{i=1}^{N} \log f^*_i(\tau_i)\right) + \log S_{N+1}^*(\tau_{N+1})$.
In the above figure this corresponds to summing up the orange entries in each row.
</figcaption>
</div>

## Putting everything together
Now we have all the pieces necessary to define and train our model.
Here is a link to the [Jupyter notebook](https://colab.research.google.com/github/shchur/shchur.github.io/blob/gh-pages/assets/notebooks/tpp2/neural_tpp.ipynb) with all the code we've seen so far, but where the different model components are nicely wrapped into a single <tt>nn.Module</tt>.

As mentioned before, we train the model by minimizing the NLL of the training sequences.
More specifically, we average the loss over all sequences and normalize it by $T$, the length of the observed time interval.

<d-code language="python">
model = NeuralTPP()
opt = torch.optim.Adam(model.parameters(), lr=5e-3)

max_epochs = 200
for epoch in range(max_epochs):
    opt.zero_grad()
    loss = model.nll_loss(inter_times, seq_lengths).mean() / t_end
    loss.backward()
    opt.step()

</d-code>

There are different ways to evaluate TPP models.
Here, I chose to visualize some properties of the event sequences generated by the model and compare them to those of the training data.
The code for sampling is not particularly interesting.
It follows the same logic as before --- at each step $i$, we sample the next inter-event time $\tau_i \sim f_i^*(\tau_i)$, feed it into the RNN to obtain the next context embedding $\boldsymbol{c}_{i+1}$, and repeat the procedure.
See the [Jupyter notebook](https://colab.research.google.com/github/shchur/shchur.github.io/blob/gh-pages/assets/notebooks/tpp2/neural_tpp.ipynb) for details.

<div class="l-body">
<img class="img-fluid rounded" src="/assets/img/posts/tpp2/visualize_results.png"
style="display: block; width: 100%; margin-left: auto; margin-right: auto;"/>
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
Comparison of real and generated event sequences.
<br>
<b>Left:</b> Visualization of the arrival times in 10 real (top) and 10 simulated (bottom) sequences.
<br>
<b>Right:</b> Distribution of sequence lengths for real (top) and simulated (bottom) event sequences.
</figcaption>
</div>

We see that training sequences have a trend: there are a lot more events in the $[0, 50]$ interval than in $[50, 100]$ (top left figure).
Our neural TPP model has learned to produce sequences with a similar property (bottom left).
The figure also shows that the distribution of the sequence lengths in real (top right) and simulated (bottom right) sequences are quite similar.
We conclude that the model has approximated the true data-generating TPP reasonably well.
Of course, there is room for improvement. After all, we defined a really simple architecture, and it's possible to get even better results by using a more flexible model.

## Concluding remarks

In this post, we have learned about the general design principles of neural TPPs and implemented a simple model of this class.
Unlike traditional TPP models that we discussed in the previous post (Poisson processes, Hawkes processes), neural TPPs can simultaneously capture different patterns in the data (e.g., global trends, burstiness, repeating subsequences).

Neural TPPs are a hot research topic, and a number of improvements have been proposed in the last couple of years.
For example, one can use a transformer as the history encoder <d-cite key="zhang2020self,zuo2020transformer"></d-cite>, or choose a more flexible parametrization of the conditional distribution <d-cite key="omi2019fully,shchur2020intensity,zhang2020cause"></d-cite>.
Some works take a completely different approach --- they directly parametrize the conditional intenisity $\lambda^*(t)$ using a hidden state that evolves in continuous time according to a (neural) ODE <d-cite key="de2019gru,rubanova2019latent"></d-cite>, instead of defining the model autoregressively.
If you want to learn more about neural TPPs, their applications, and open challenges, you can check our recent survey paper <d-cite key="shchur2021neural"></d-cite>.

So far we have been talking exclusively about the so-called *unmarked* TPPs, where each event is represented only by its arrival time $t_i$.
Next time, I will talk about the important case of *marked* TPPs, where events come with additional information, such as class labels or spatial locations.


### Acknowledgments
I would like to thank [Daniel Zügner](https://twitter.com/DanielZuegner) for his feedback on this post.
