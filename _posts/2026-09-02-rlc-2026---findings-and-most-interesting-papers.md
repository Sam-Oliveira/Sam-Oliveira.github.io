---
layout: post
title: "RLC 2026 - Findings and Most Interesting Papers"
date: 2026-09-02 09:00:00-0600
description:
tags:
related_posts: false
---

<img src="{{ site.baseurl }}/assets/img/blog/rlc-2026---findings-and-most-interesting-papers/Pasted image 20260823102317.png" width="415" style="max-width:100%;height:auto;">

I had the opportunity to attend RLC this year, in Montreal, and decided to highlight some of the works I found the most interesting (mostly related to Continual Learning), as well as some general trends I see in the field.

## Contents
- [Continual Learning Workshop](#continual-learning-workshop)
- [Main Conference](#main-conference)
- [Overall trends](#overall-trends)

## 1. Continual Learning Workshop {#continual-learning-workshop}

#### 1.1. [Task Diversity Produces Systematic Transfer but Inhibits Continual Reinforcement Learning (Seth et al.)]([https://openreview.net/forum?id=3WyLUImqng](https://openreview.net/pdf?id=3WyLUImqng))

This paper asks whether the distribution of tasks impacts the performance of a continual RL agent, in particular whether task diversity allows for the agent to keep learning across distribution shifts. 

**Environment.**

<img src="{{ site.baseurl }}/assets/img/blog/rlc-2026---findings-and-most-interesting-papers/Screenshot 2026-08-23 at 10.43.52 AM.png" width="405" style="max-width:100%;height:auto;">

To help study its research question, the paper introduces Banyan, a GPU-accelerate domain for continual RL with procedurally-generated tasks/envs. The task space is compositional, so the paper can better study how the amount and type of task diversity affects CRL, over very large sequences of tasks.

Tasks are based on a mini-grid agent that must combine objects (over multiple steps) in a maze to create a goal object. A task is specified by a *task tree* (specifies which objects must be combined to produce a goal object), and a *layout* (the maze configuration and location of objects). This allows for three axes of task diversity: layouts, task-tree topologies, and task-tree instances (topology is the same but we change the objects). Changing tree topologies in particular changes the way in which sub-goals must be composed (and their order), thus changing a subgoal's policy. Changing object assignment preserves the structure of a solution, while requiring it to adapt to new objects. The agent observes the entire grid and a description of the goal object (it does not know the task tree structure). The agent can move in four directions and "stay", and has three actions 1) to pick up objects; 2) to drop them to combine them with others; 3) to transform a single object. 

Importantly, how specific objects combine is fixed across tasks, and thus is knowledge that the agent can re-use across tasks: there is a sort of object vocabulary where the agent can learn the specific way in which objects or their pairings work.

**Findings.** Task diversity improves forward and backward transfer for a single shift (training on a diverse distribution, and evaluating transfer to a different distribution). However, when considering a sequence of distribution shifts (it is unclear from the main text how these distributions are different), **initial performance on the new task improves but learning (end performance) is hindered**, that is the agent plateaus, even when combining PPO with Continual Backprop. This is very similar to the [conclusion](https://arxiv.org/abs/2607.17326) my research group reached on zero-shot transfer, albeit there is no learning in that setting.

There is thus a sort of **generalization-specialization trade-off**, where the agent learns the general task structure due to the diversity of tasks it has faced, but struggles to specialize in each one task. 

An obvious question here is whether these results show us something more intrinsic to continual learning, or whether they simply reflect the limitations of current CL algorithms. My intuition tells me this is largely the latter. In particular, while the authors evaluate a variety of reinitialization algorithms (e.g. Continual Backprop), at no point do they consider RL algorithms specifically built for CRL. Instead, only PPO and PQN are attempted. There is thus no inherent compositionality or a hierarchy to the learning algorithm, and thus it cannot exploit the structure of the task tree.

Furthermore, I think these results need to be taken with a grain of salt. Most of the results are for a single distribution shift, which doesn't actually reflect continual learning settings. It would be interesting to study the different types of task shifts (layout, object and topology) when considering an actual sequence of task distributions, as well as study loss of plasticity metrics to investigate whether CBP is failing to contain this. Most importantly, Figure 6 doesn't evaluate all methods in the same way: the $256^2$ curve for example sees $256^2 * 10$ tasks in total, and at each point in sequence is being evaluated on its performance on $256$ tasks, whereas $1^2$ is only being evaluated on one task at a time. This seems to not be a fair comparison: we do indeed study the learning dynamics, but those don't necessarily reflect how "good" the agent is.

Finally, this study only uses recurrent architectures, although the authors do not specify the exact architecture. It is entirely possible the large ($10^9$) number of env steps is just too much for such an architecture, or that CBP does not perform well for it.  

<!-- Ideas for extended work:
- study loss of plasticity neural network metrics under these settings
- figure out what distributions they actually use, and how we can alter this to better study what is happening -> what does the 256^2 in fig 6 actually means? it is being trained on 256^2 tasks across each task in the sequence??! Apparently yes: "We co-vary layout and topology diversity from n = 1 to n = 256, yielding up to n2 = 65,536 depth-6 layout–topology combinations per distribution"
- they study this under recurrent architectures!!! Asdk them about what actual architecture they use and why, and if CBP has been used with it before
--> 
#### 1.2. [Calibrated Partial Resets: Preventing Policy Collapse in Continual Reinforcement Learning (McCutcheon et al.)](https://openreview.net/pdf?id=OJA7FmLMCN)

This was perhaps my favourite paper of the entire conference, as it touches specifically on reinitialization algorithms for continual learning.

In typical Generate and Test algorithms (or reinitialization algorithms, whatever you prefer to call them), either neurons or individual network parameters are generated, evaluated/tested based on a utility function, and then a specific portion of them get re-generated. Works so far (CBP, ReDo, SWR) simply reinitialize/reset the neuron (or the weight ,depending in what space they work in) by settings its value to that of some initial distribution (this is slightly more complicated in the terms of neurons). However, such a reset is highly destructive, as it fully removes any info currently present in that parameter or neuron. 

This work proposes an alternative, Calibrated Partial Resets, where neurons are periodically pushed towards their initialization, but only partly, with the strength of such a pull depending on their utility. 

Thus, CPR retains the selectivity of neuron-reset methods while avoiding the all-or-nothing updates that can make full reinitialization brittle in continual RL. In this way, it differs from full reinitialization methods in two ways: (1) it doesn't re-initialize a fixed set of neurons, instead it (partly) reinitializes all neurons, with the strength of such reinitialization being dependent on the utility; (2) this reinitialization is not full towards the initialization value, but instead only partial (an average between the current weight and its initialization).

**Method: Calibrated Partial Resets (CPR).** It works as follows. For neuron $i$ in layer $\ell$, utility is the mean gradient magnitude of its incoming weights:

$$S_i^\ell(t) = \mathbb{E}_{x \in \mathcal{B}_t}\left[\left\|\nabla_{W_{i,:}^\ell}\mathcal{L}(x)\right\|\right]$$

$S_i^\ell$ is smoothed with an EMA and divided by the layer's mean score:

$$u_i^\ell(t) \leftarrow \beta\, u_i^\ell(t-1) + (1-\beta)\,\tilde{S}_i^\ell(t), \qquad \tilde{S}_i^\ell(t) = \frac{S_i^\ell(t)}{\frac{1}{n_\ell}\sum_k S_k^\ell(t)}$$

Then, a monotonically decreasing sigmoid-based function turns utility into a $[0,1]$ intensity:

$$\phi(u_i^\ell) = \min\!\big(2\sigma[-\kappa(u_i^\ell - 1)],\, 1\big)$$

- units at or above the layer-average utility ($u \geq 1$) get $\phi \approx 0$
- units well below average get $\phi \to 1$
- $\kappa$ controls how sharp the transition is (fixed at $\kappa = 16$ in the paper)
This intensity is then scaled by $\rho \in (0,1]$, the maximum fraction any single neuron is allowed to be pulled toward reinitialization, to get the actual per-neuron reset coefficient:

$$r_i^\ell = \rho\,\phi(u_i^\ell) \in [0, \rho]$$

Low-utility units get $r_i^\ell$ near $\rho$ (strong pull); high-utility units get $r_i^\ell$ near 0 (left alone).
Every $f$ steps, each unit's incoming and outgoing weights are linearly interpolated towards a fresh random draw $\xi_{i,:}^\ell \sim \mathcal{D}_{\text{init}}$ (e.g. Kaiming/Xavier), by that unit's own coefficient:

$$W_{i,:}^\ell \leftarrow (1-r_i^\ell)\,W_{i,:}^\ell + r_i^\ell\,\xi_{i,:}^\ell$$

$$W_{:,i}^{\ell+1} \leftarrow (1-r_i^\ell)\,W_{:,i}^{\ell+1}$$

<img src="{{ site.baseurl }}/assets/img/blog/rlc-2026---findings-and-most-interesting-papers/Screenshot 2026-08-23 at 7.16.59 PM.png" width="592" style="max-width:100%;height:auto;">

**Results.** The results are quite impressive: CPR is the only method to avoid policy decay in slippery Ant over 400M training steps. Other reinitialization methods either seem to eventually collapse, or are stable but converge to a suboptimal policy. 

They also test the method on Continual World and Continual MinAtar. In Continual World, CPR seems to outperform reinitialization methods, but all these methods (including CPR) seem to be very far in terms of performance compared to some of the methods in the Continual World leaderboard. The authors also show that CPR is as successful as CBP in keeping the percentage of dormant or linearized neurons stable.

Finally, the authors discuss how $\rho$ acts as a control parameter over the stability-plasticity trade-off, determining the amount of reinitialization that neurons are allowed to undertake. 

#### 1.3. [Connectivity, Credit Assignment and the Speed of Learning](https://openreview.net/pdf?id=JqQZ18oZqs)

This paper studies how a MLP connectivity affects its speed of learning in stationary and non-stationary settings. I will not get into the experimental details too much, but the authors design a task (with a stationary and a non-stationary variant) where a block network structure is optimal, and then compare its learning performance to a dense network. There are two main findings:
- Fully connected architectures suffer from interference problems when propagating gradients, but such problems mostly affect the speed of learning and thus are not represented in the asymptotical performance often reported.
- Starting from a block network that has converged to the solution, and adding weights set to 0 to make the network dense and training that dense network leads to worse performance than the original block network. The authors posit this is due to worse credit assignment through the network, as part of its updates are performed on weights that do not relate to the task.
It is a nice read, and it motivates some of the work I am interested in regarding learning the connectivity of neural networks.

## 2. Main Conference {#main-conference}
#### 2.1. [The Cell Must Go On: Agar.io for Continual RL](https://arxiv.org/pdf/2505.18347)

This paper introduces AgarCL, a continual RL environment based on the game of Agar.io. Most interestingly, the environment is non-episodic and introduces gradual non-stationarity, instead of the usual sharp non-stationarity in continual learning benchmarks. In particular, the game is never-ending, and the environment dynamics (both the observations and the consequences of actions) change as the agent's mass changes, and thus the agent needs to constantly adapt. 

The authors show both typical RL algorithms like PPO/SAC, as well as PPO with Continual Backprop and ReDo fail. However, the latter's hyperparameters were not tuned, which makes it hard to conclude whether such reinitialization algorithms with PPO they truly fail in this environment. 

While these types of environments are still game-based and thus very simplistic, I think this could be a very interesting benchmark in which to test the long-term performance of CRL algorithms, as well as their performance with respect to much smoother non-stationarities (which are believed to be less prone to cause issues like loss of plasticity). 

At the same time, it is still ultimately the same task, with the same reward function, and thus I struggle to see how this environment leads to conflicting information/gradients which are often destructive in continual learning. Furthermore, there is not necessarily a large amount of forward transfer needed: future tasks might as well be able to be solved without previous information. The negative empirical results of reinitialization algorithms are interesting, showing that it is still indeed a challenging baseline, but I'd be interested to tune the hyperparameters before concluding anything else.

#### 2.2. [Revisiting Adam for Streaming Reinforcement Learning](https://arxiv.org/abs/2605.06764)

This work studies the impact of (1) the Adam optimizer, and (2) objectives with bounded gradients on deep streaming RL performance. In particular, the authors study DQN and C51 and show that such algorithms can (with the normalization techniques of stream-x) perform reasonably in streaming settings with Adam due to the derivative of their objective being bounded, and weight updates being variance-adjusted.

In particular, they show DQN (without eligibility traces) without replay buffers can perform well, if we 1) let it have the normalization techniques of stream-x and stuff like LayerNorm; and 2) we tune the Adam hyperparameters. Thus, even without the fancy optimizer of stream-x (and without eligibility traces), DQN without replay buffers can work just as well, although this depends quite strongly on the environment.

They then propose their own version of $Q(\lambda)$, called Adaptive $Q(\lambda)$, that uses eligibility traces that are normalized by an EMA of the squared gradients of the state-action value function $Q$ (while Adam keeps an EMA of the gradient and normalizes it by an EMA of the squared gradients, $AQ(\lambda)$ instead keeps an accumulating eligibility trace but normalizes it by an EMA of the squared value function gradients). Their method also requires clamping the TD error.
	<img src="{{ site.baseurl }}/assets/img/blog/rlc-2026---findings-and-most-interesting-papers/Screenshot 2026-09-02 at 10.48.30 AM.png" style="max-width:100%;height:auto;">
It is not clear how the update of $AQ(\lambda)$ compares with the ObGD optimizer used in the stream-x family, and whether ideas from both could be combined. They both clip the TD error. However, while ObGD only adjusts the step size (keeping the eligibility trace direction fixed), $AQ(\lambda)$ inherently changes the eligibility trace direction by scaling components according to the EMA of their squared value. It is unclear to me whether this idea of parameter-wise scaling makes as much sense in an eligibility trace set-up, but it is interesting nonetheless.

#### 2.3. [WiSCA: Winning Subnetworks with Cross-Task Attention](https://openreview.net/pdf?id=v7XWnkIWwq)
This paper introduces WiSCA, a new parameter isolation method for continual reinforcement learning. It builds on top of Winning Subnetworks (WSN). WSNs learns per-task binary masks over a network's parameters. Importantly, parameters used by previous tasks can be used by the current task, but cannot be altered. This ensures no-forgetting, as each task's performance can simply be retrieved based on the current network and the task's respective binary mask. 

Critically, while this allows for no forgetting, forward transfer is rather limited, as it can only appear from a task using a previous task's fixed parameter. WiSCA aims to improve the forward transfer capabilities of Winning Subnetworks by introducing per-task attention over the policy logits of all previous tasks. In this way, while isolating task-specific parameters within an encoder network, the actor can then, through a learnt attention mechanism, learn to combine previous policies. 

This is a really cool idea, particularly because it scales better than previous attention-based approaches like CompoNet (since no new encoder is added per task, and instead only small adaptation heads). However, it still requires the addition of a per-head task (among other "light" components), relies on task identity, and is limited to the actor network: the critic network simply has a new head added per task.

#### Other interesting work
- [Gradient Iterated Temporal-Difference Learning](https://arxiv.org/pdf/2603.07833)
- [A Survey of State Representation Learning for Deep Reinforcement Learning](https://arxiv.org/pdf/2506.17518)
- [Cohering Reinforcement Learning](https://rlj.cs.umass.edu/2026/papers/Paper138.pdf) (by far the "weirdest" and most thought-provoking paper I saw at RLC)

## 3. Overall trends (particularly regarding Continual Learning) {#overall-trends}

### 3.1. A field reluctant to face what continual learning is supposed to be

One of RLC's outstanding paper awards went to [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://arxiv.org/pdf/2603.11653). As has been common in recent years, this work makes quite exaggerated claims regarding the continual learning capability of large models (in this case VLAs). The paper's central claim (that VLAs sequentially fine-tuned with on-policy RL using LoRA can continually learn) is well supported within the regime that the paper tests, but that regime is once again out of touch with what continual learning actually entails. 

A large majority of the paper focuses on sequences of 5 (!) tasks from libero-spatial, -object, -long-horizon that largely share the desired behaviour, varying only the target object or a spatial descriptor, with task identity handed to the policy via language conditioning rather than inferred. Almost the entire paper focuses on sequences of 5 tasks(!). The one experiment that considers a longer sequence of tasks (30) is relegated to the supplementary material, and is presented without any information regarding task names, number of seeds or uncertainty values, with performance not even being compared against any baselines.

A second related limitation relates to the pre-trained VLA's competence on the tasks: held-out tasks are drawn from the same object/task family as the training tasks, and the base VLA already succeeds 55–87% of the time on training tasks before training even begins. As a matter of fact, the paper explicitly removes tasks for which performance is near 0!). Thus, not only does the paper only consider the same style of tasks where there is no conflicting knowledge between tasks, but the model already possesses a quite large success rate on the tasks, and thus does not acquire any new skills from scratch.

Obviously the paper still holds a valuable empirical conclusion, particularly that (as in LLMs) on-policy RL seems to be more resistant to catastrophic forgetting, and that LoRA with GRPO is a somewhat promising direction for sequential fine-tuning. However, "VLAs are Continual Learners" is a much broader claim than what any of this experimental setup was designed to show. This recent wave of papers over-claiming the continual learning capabilities of large models seems very dangerous and unproductive to me, inciting a false sense of security within the field, while the underlying continual learning problem remains unsolved (and understudied).
### 3.2. Benchmarks: perhaps still too gamified?
There were a few works proposing new benchmarks for Continual RL, including the already mentioned [Banyan](https://arxiv.org/abs/2606.00880) and [AgarioCL](https://arxiv.org/pdf/2505.18347), but also [Forager](https://arxiv.org/pdf/2605.01131). While these works are all very valuable, and I am happy to see the community directly working on benchmarks, I can't help but think that we are still missing something when it comes to CRL benchmarks. In particular, from my understanding none of these benchmarks create tasks that are only solvable if the agent has solved previous tasks, thus requiring the agent to compose knowledge if it wishes to solve the task. Obviously this capability will make agents perform better in the benchmarks mentioned above, but it is not clear to me that it NEEDS that capability to solve the tasks. It would be interesting to see the RL community consider design decisions like those made for [Continual Learning Bench](https://continual-learning-bench.com/), where agents need to explicitly remember past information to be able to solve future tasks correctly.
