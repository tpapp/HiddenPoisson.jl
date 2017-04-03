# HiddenPoisson

[![Project Status: WIP - Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.org/tpapp/HiddenPoisson.jl.svg?branch=master)](https://travis-ci.org/tpapp/HiddenPoisson.jl)
[![Coverage Status](https://coveralls.io/repos/tpapp/HiddenPoisson.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/tpapp/HiddenPoisson.jl?branch=master)
[![codecov.io](http://codecov.io/github/tpapp/HiddenPoisson.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/HiddenPoisson.jl?branch=master)

## Overview

Simulate a hidden state model in continuous time with the following setup:

1. Let `state` denote the hidden (unobserved) *state*. It can be an arbitrary Julia object, as long as the relevant methods are defined.
2. The *model* parameters are packed into the `model` object.
3. `rates(model, state)` returns a vector of Poisson rates `λᵢ`. Each *shock* `i` can happen at rate `λᵢ` independently (competing Poisson processes).
4. When an `i` is realized, `next_state(model, state, i)` is used to obtain the next state (which, again, can be an arbitrary object). This function may be stochastic.
5. `observation(model, state)` is another (possibly stochastic) function for returning an observation for the hidden `state`.

The user needs to define a model structure, then
```julia
import HiddenPoisson: rates, next_state, observation
```
and then define these methods for the model structure. Then simulations are obtained using `next_observation` or `next_observations`. See the tests for examples.

## How it works

A `HiddenState` object keeps track of the time remaining until the next shock, and its value. Given a time interval `t`, new transitions are generated on demand until enough time has passed.
