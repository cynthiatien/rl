In this repository, I will share my knowledge and works about reinforcement learning. I will write blog posts and implement the algorithms in order to understand them well. I will also share some articles that will help to understand the concepts better.

## Algorithms

#### Vanilla Policy Gradients(REINFORCE)

[Check out the blog post for detailed explanation.](https://snnclsr.github.io/2019/03/14/policy-gradients/)

**Summary:** Policy gradient algorithms directly learn/optimize the policy. We generate samples from the environment. We calculate the sum of gradients along the samples and, also we compute the total reward for each sample. We multiply them and optimize with gradient ascent.

[Code](https://github.com/snnclsr/rl/blob/master/vpg.py)

**Skeleton code for the implementation is taken from Berkeley RL Course Assignment 2 which can be found [here](https://github.com/berkeleydeeprlcourse/homework). Also check out the course content from [here](http://rail.eecs.berkeley.edu/deeprlcourse/).**
