# CodeNames Debate

[CodeNames](<https://en.wikipedia.org/wiki/Codenames_(board_game)>) is a party card game where players need to find creative word associations. It has some properties that make it compelling as a testbed for scalable oversight experiments:

* It should be easy for language models to learn.
* The computational complexity for generating a clue is much larger than for finding a flaw in a clue, which is higher still than evaluating a flaw.
* It's easy to procedurally generate many games.
* It's relatively easy to simulate overseers with various kinds of flaws, or artificially limit the oversight budget.

This project aims to expand on the theory of predicting whether a scalable oversight technique will robustly succeed for some problem domain and overseer, and then test out the theory with many small experiments.

For more detail, see [the project doc](https://docs.google.com/document/d/1t9sBDuqqvJ4FetyBXfnLMVyoYAL2tDbddS-kJAHifqs/edit?usp=sharing).

## Relevant background
* [the original Debate paper](https://arxiv.org/abs/1805.00899)
* Redwood Research's post on [meta-level adversarial evaluations](https://www.alignmentforum.org/posts/MbWWKbyD5gLhJgfwn/meta-level-adversarial-evaluation-of-oversight-techniques-1)

## Rough Roadmap
(Not really kept up to date, for full details check out the project doc)

- [x] Generate decent clues with GPT-4
- [x] Fine tune an open source LLM on the GPT-4 clues
- [x] Get GPT-3.5 to play as the receiver of the clue (ground truth clue reward)
- [x] POC of improving clue quality with DPO
- [x] Implement "RLHF" using a simulated overseer
- [x] Implement Debate/Critiques (they look pretty much the same here)
- [x] Insert a variety of flaws into the overseer
- [x] Add a meta-level adversarial incentive
- [ ] Generate a huge matrix of experiment parameters and run all of them
