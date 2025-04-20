# CodeNames Oversight

This repo contains the code for [Evaluating Oversight Robustness with Incentivized Reward Hacking](https://www.lesswrong.com/posts/gHheKspmTAYqEagbT/evaluating-oversight-robustness-with-incentivized-reward).

[CodeNames](<https://en.wikipedia.org/wiki/Codenames_(board_game)>) is a party card game where players need to find creative word associations. It has some properties that make it compelling as a testbed for scalable oversight experiments:

* It should be easy for language models to learn.
* The computational complexity for generating a clue is much larger than for finding an issue with a clue, which is higher still than evaluating an issue.
* It's easy to procedurally generate many games.
* It's easy to simulate overseers with various kinds of flaws, or artificially limit the oversight budget.

This project aims to expand on the theory of predicting whether a scalable oversight technique will robustly succeed for some problem domain and overseer, and then test out the theory with many small experiments.

## Relevant background
* [the original Debate paper](https://arxiv.org/abs/1805.00899)
* Redwood Research's post on [meta-level adversarial evaluations](https://www.alignmentforum.org/posts/MbWWKbyD5gLhJgfwn/meta-level-adversarial-evaluation-of-oversight-techniques-1)
