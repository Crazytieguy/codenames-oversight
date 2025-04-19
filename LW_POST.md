# CodeNames Oversight (Name TBD)

## Introduction

### Scalable oversight research paradigm

Scalable oversight is the challenge of overseeing or training AI systems to achieve goals that are hard for humans to specify. By definition, studying scalable oversight directly on the tasks we care about is infeasible, because we would not know if the goals were achieved correctly. So instead we use sandwiching - using an (artificially) week/limited overseer to oversee a model to achieve a goal that a more powerful/reference overseer can measure. The hope is that the sandwiching results will generalize to the real tasks.

### Our high level contributions

We identified several new approaches to sandwiching, and implemented them in a controlled and simplified setting as a proof-of-concept. Out hope is that our experiments can inform future scalable oversight research, helping with generalization to advanced AI:

- **Incentivized reward hacking (link redwood post on meta-level-adversarial robustness)**
  
  A successful sandwiching experiment usually represents just one example of the training technique working - but it could be sensitive to the choice of hyperparameters, model architecture, etc. In order to improve the robustness of the result, we modify the reward function to scale inversly with the reference overseer's reward, incentivizing the model to trick the overseer. This incentive significantly decreases the chance of the technique seeming to work, while the overseer is in fact exploitable.

- **Analysis of oversight techniques covering for overseer flaws.**
  
  The current scalable oversight literature seems to assume that results for a technique on some task will generalize to other tasks and other overseers. In our opinion, claims that results will generalize should be accompanied with an analysis of the mechanism by which the technique covers for the the overseer's weeknesses in overseeing the task, and an argument for the same kind of mechanism working analogously in the deployment task.

- **Training a model via RL to accomplish a fuzzy task.**

  Most sandwiching experiments so far tested whether a technique helps an overseer achieve the correct judgement at inference time without training the model, and mostly focus on multiple choice questions. We wanted to push on this frontier by attempting to actually train an LLM to accomplish a fuzzy task using scalable oversight (link to new jan leike post). This presented new challenges that we had to get creative to solve, such as designing a novel reward function.

### High level experiment and results overview

We implement a synthetic CodeNames environment where we can generate an arbitrary amount of training data, simulate overseers with a variety of flaws, and implement a variety of scalable oversight techniques. We are able to demonstrate that scalable oversight techniques can fully compensate for different overseer flaws and still train the model to a similar level of performance as a non-scalable technique with a robust overseer, albeit with greater compute requirements (this can be seen as alignment-tax). We also train the model with different amounts of adversarial incentive and show that the amount of incentive needed to elicit reward hacking is dependant on the training protocol, with the scalable oversight techniques requiring more adversarial incentive than the base technique. Finally, we implement a classical search function in place of the LLM and learn that it reward hacks more than the LLM for a similar amount of optimization strength and adversarial incentive, suggesting an inductive bias in favor of the true goal in the LLM and in the scalable oversight techniques.

## Experiment environment

In order to realistically run a large matrix of experiments with different scalable oversight techniques, adversarial incentives, and overseer flaws, we chose the task of giving the first clue in the game CodeNames.

![Game Diagram](diagrams/game-diagram.png)

In this experiment, Player 2 is a static embedding model, making guesses based on cosine similarity. Player 1 is a model being trained with RL via an overseer. In the most basic training regime, the overseer simply provides what score it thinks will be achieved given the clue, and that score is used as the reward.

![Base Protocol](diagrams/base-protocol.png)

### How we simulate overseers and their flaws

Each overseer is a python function that takes in the game and the model outputs, is able to make queries to the embedding model (guesser), and outputs a score or judgement. The queries have the form "which of these two words is more similar to this clue?". There is one non-flawed overseer that implements a correct algorithm for determining the score/judgement, and several flawed overseers:

- Lazy overseer: Stops short of iterating over all of the good words. Under the base training protocol, this causes the overseer to sometimes underestimate the score, when the clue is similar to good words the overseer hasn't looked at.
- Negligent overseer: Stops short of iterating over all of the bad words. Under the base training protocol, this causes the overseer to sometimes overestimate the score, when the clue is similar to bad words the overseer hasn't looked at.
- Biased overseer: When making a query to the embedding model, it multiplies the cosine similarity of some of the bad words by a constant factor less than 1, meaning they underweigh them. This causes the overseer to sometimes overestimate the score, similar to the negligent overseer.

### Scalable Oversight Training Protocols

![Consultancy Protocol](diagrams/consultancy-protocol.png)

In the consultancy protocol, the model provides a list of words it's targetting with the clue, and the overseer just needs to decide whether they think all of the targets will be picked by the guesser. The target words are analogous to arguments defending the output, thus delegating argumentation work from the overseer to the model. This protocol covers for overseer laziness, because the overseer is not required to iterate over the good words.

![Critiques Protocol](diagrams/critiques-protocol.png)

In the critiques protocol there is a second model that is trained in tandem with the first to give critiques (this bad word will be picked before this target). The overseer just needs to decide whether at least one critique is valid. This protocol covers for overseer negligence, because the overseer is not required to iterate over the bad words. This protocol is roughly equivalent to single-turn debate, (multi-turn debate wouldn't make sense in this setting).

Note: Neither protocol covers for bias. Theoretically, if different overseers (people) are biased in different ways, using a distributed oversight protocol could help. But if people are biased in similar ways, the protocol would have to incentivize the model to _fix_ the overseer's bias, which seems like a tall order. Thus we find it important to identify how humans are biased in their final judgements, while humans' flawed ability to come up with things to judge can be expected to be covered by scalable oversight techniques.

TODO: further discussion of human flawes with examples.

### Reward function

For the base protocol the reward function is sraightforward - just the overseer's prediction of the score. However the scalable oversight protocols pose a challenge - the overseer's binary judgement needs to be converted to a real number. The naive approach would be issuing a reward equal to the number of targets if the overseer accepted them, and 0 otherwise. However, this would mean that adding targets becomes less worthwhile for the model the more targets there are - it risks losing an increasing amount of reward for a constant amount of reward. Experiments that used this approach in fact resulted in conservative models that only target 1 word. The approach that ended up working here is quite complicated, but can be lossily summarized as scaling the reward exponentially with the number of targets and adjusting the base on the fly during training to ensure the model is always incentivized in the right direction. (link to appendix for full derivation)

A simpler alternative we didn't try is to sample two outputs and give them 1 and -1 according to some ordinal rule.

### Adversarial Incentive

In order to better differentiate between robust and exploitable protocol+overseer pairs, we modify the reward function to include an adversarial incentive:

$R = R_o + \alpha (R_o - R_r)$

Where $R_o$ is the reward of the overseer, $R_r$ is the reward of the reference overseer, and $\alpha$ is the adversarial incentive strength. This formulation has two desireable properties:

- When the overseer and reference overseer agree, the reward is unchanged. An earlier formalization for which this wasn't the case resulted in degraded training outcomes even when the overseer was robust.
- It accomodates the weird reward function by utilizing a reference overseer rather than the direct ground truth score. Using the grount truth again resulted in unnecesarily degraded performance.

It's worth noting that under the critiques protocol, we use the robust consultancy overseer as the reference overseer. This leaves in the exploitability of the protocol due to inaccuracies of the critique model.

### Training

All of the models are parameter-efficient fine tunes of llama-2 7b.

#### Initial fine tuning

All models were supervised-finetuned on random outputs that follow the correct format. Using none-random outputs (such as a dataset generated from another model) may limit the creativity of the RL training, or bias the initial model with respect to the different overseers and protocols. (appendix with full details)

#### RL training

We used a slightly modified version of the RLOO implementation from huggingface trl (link). RLOO Proved much more stable and efficient than every version of PPO we tried, and was simple enough to make it feasible to modify. (appendix with full details)

## Results

### The Scalable Oversight Techniques Work, but Require More Compute

Here are some plots showing true reward and overseer reward over training for the different protocols, with the robust overseer.

![Base Robust Training](results/plots/base-robust-training.png)

For the base protocol the two lines are the same - the robust overseer simply gives the true score. The score reaches 2.8 within ~53 training steps, and increases to ~3.29 by the end of the 336 training steps.

![Consultancy Robust Training](results/plots/consultancy-robust-training.png)

For consultancy the overseer estimated score is lower than the true score, due to the targets mechanism. The true score reaches 2.8 within ~238 steps, and increases to ~3.11 by the end of the 672 training steps.

![Critiques Robust Training](results/plots/critiques-robust-training.png)

For critiques the overseer estimated score starts out higher than the true score due to the random critiques, but quickly decreases to bellow the true score. The true score reaches 2.8 within ~290 steps, and reaches ~2.95 by the end of the 672 training steps. Note that a critique training step requires roughly 2x the compute as a consultancy training step, due to the additional model.

#### Oversight Efficiency

Although the scalable oversight techniques require more compute, they also require less oversight work per training step. Specifically, the base protocol with robust overseer used on average $8.96$ comparisons per clue, the consultancy protocol used $4.52$ comparisons, and the critiques protocol used $2.25$ comparisons. So the base and consultancy runs used roughly the same amount of oversight work, and the critiques runs used about half as much.

### Adversarial Incentive Elicits Quite Reward Hacking

TODO: explain the weird plot format

![Oracle Adversarial Strengths](plots/oracle-adversarial-strengths.png)

The classical search (oracle) experiments show how much reward hacking we should expect from different amounts of optimization and adversarial incentive. Each colored line represents an optimization strength, each grey line represents and adversarial incentive strength, and each intersection represents a single experiment. As optimization strength increases, so does the distance between the overseer reward and true reward. Additionally, the policy is able to find worse outputs that don't decrease the overseer reward (the vertical sections).

![Base Adversarial Strengths](plots/base-adversarial-strengths.png)

The base protocol experiments look like somewhere between optimization strength 8 and 16, except they seem somewhat biased in favor of the true reward, presumably due to some sort of inductive bias. An adversarial strength of 0.375 causes a ~0.65 reduction in true score while only causing a ~0.05 reduction in overseer score.

![Consultancy Adversarial Strengths](plots/consultancy-adversarial-strengths.png)

Eliciting reward hacking under the consultancy (and critiques) protocol was much harder, requiring a stronger flaw and a greater adversarial incentive, suggesting an even stronger inductive bias in favor of the true goal. An adversarial strength of 0.75 causes a ~0.2 reduction in reference overseer score while causing a ~0.05 reduction in overseer score. Note - these numbers are not directly comparable to the oracle and base protocol numbers, due to the consultancy overseer underestimating the score.

### Scalable Oversight Covers for Overseer Flaws

### Qualitative Policy Analysis
