# FoRL

A library for for First-order Reinforcement Learning algorithms.

In a world dominated by Policy Gradients based approaches, we have created a library that attempts to learn policies via First-Order Gradients (FOG). Also known as path-wise gradients or the reparametarization trick. Why? FOGs are knwon to have lower variance which translates more efficient learning but they are also don't perform very well with dicontinuous loss landscapes.

Applications:

- Differentiable simulation
- Model-based RL
- World models

## Installation

Tested only on Ubuntu 22.04. Requires Python, conda and an Nvidia GPU with >12GB VRAM.

1. `git clone git@github.com:pairlab/FoRL.git`
2. `cd FoRL`
3. `conda env create -f environment.yaml`
4. `ln -s $CONDA_PREFIX/lib $CONDA_PREFIX/lib64` (hack to get CUDA to work inside conda)
5. `pip install -e .`

## Examples

### Dflex

One of the first differentiable simulations for robotics. First proposed with the [SHAC algorithm](https://short-horizon-actor-critic.github.io/) but is now depricated.

```
cd scripts
conda activate forl
python train_dflex.py env=dflex_ant
```

The script is fully configured and usable with [hydra](https://hydra.cc/docs/intro/).

### Warp

The successor of dflex, warp is Nvidia's current effort to create a universal differentiable simulation.

TODO examples

## Gym interface

We try to comply with the normnal [gym interface](https://www.gymlibrary.dev/api/core/) but due to the nature of FOG methods, we cannot do that fully. As such we require gym envs passed to our algorithms to:

- `s, r, d, info = env.step(a)` must accept and return PyTorch Tensors and maintain gradients through the funciton
- The `info` dict must contain `termination` and `truncation` key-value pairs. Our libtary does not use the `d` done flag.
- `env.reset(grad=True)` must accept an optional kwarg `grad` which if true resets the gradient graph but does not reset the enviuronment

[Example implementation of this interface](https://github.com/imgeorgiev/DiffRL/blob/main/dflex/dflex/envs/dflex_env.py)

## Current algorithms

* [Short Horizon Actor Critic (SHAC)](https://short-horizon-actor-critic.github.io/)

## TODOs

- [x] Upgrade python version
- [x] Vectorize critic
- [x] Try Mish activation - helps
- [x] Try stop gradient on actor - hurts 
- [x] Try regressing values - hurts
- [ ] Try return normalization
- [ ] Verify safe/load
- [ ] Think about simplified gym interface that is compatible with rl_games
- [x] More dflex examples
- [ ] Add warp support
- [ ] Add AHAC algorithm
