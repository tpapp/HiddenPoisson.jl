module HiddenPoisson

using ArgCheck
using Distributions
using Parameters
using StatsBase

export
    transitions, observation, # users define methods for these
    HiddenState, next_observation, next_observations # simulation interface
    
"""
```julia
transitions(model, state)
```

Return a vector of `Pair(λ, next_state)` elements, where the `λ`s are Poisson
rates for shocks, while `next_state` is either a closure that generates a new
state, or a deterministic value.

The user needs to supply a method for a given `model`.
"""
function transitions end

"""
```julia
observation(model, state)
```

Draw a (possibly random) observation of the `model` in `state`.

The user needs to supply a method for a given `model`.
"""
function observation end

"""
```julia
competing_poisson(rates)
```
Draw a random first arrival time for competing Poisson processes with `rates`.

Return `T, i` where

- `T` is the arrival time and
- `i` is the index of the event which happened first.
"""
function competing_poisson(rates::AbstractVector)
    first_time = rand(Exponential(1/sum(rates)))
    index = sample(indices(rates, 1), WeightVec(rates))
    first_time, index
end

"""
T, next_state = random_transition(trans)

Given a the object returned by `transitions(model, state)`, draw a random
stopping time and next state generator, and return these.
"""
function random_transition(trans)
    T, i = competing_poisson(first.(trans))
    T, trans[i].second
end

random_transition(model, state) = random_transition(transitions(model, state))

"""
Representation of a hidden Poisson process.

`model` contains the model parameters, and is used by `rates` and `observation`.

`state` is the current state (an arbitrary object).

`T` is the time left until the next Poisson event, which is simulated using `rates`.

`next_state` is the function that generates the new state at `T`.
"""
immutable HiddenState{Tm, Ts, Tt <: Real, Tn}
    model::Tm
    state::Ts
    T::Tt
    next_state::Tn
    function HiddenState(model, state, T, next_state)
        @argcheck T ≥ 0
        new(model, state, T, next_state)
    end
end

HiddenState{Tm,Ts,Tt,Tn}(model::Tm, state::Ts, T::Tt, next_state::Tn) =
    HiddenState{Tm,Ts,Tt,Tn}(model, state, T, next_state)

function HiddenState(model, state)
    HiddenState(model, state, random_transition(model, state)...)
end

"""
```julia
o, new_hs = next_observation(hs, t)
```

Return observation `o` after time `t` starting from hidden state `hs`, and the
new hidden state.
"""
function next_observation(hs::HiddenState, t::Real)
    @argcheck t ≥ zero(t)
    @unpack model, state, T, next_state = hs
    while t ≥ T
        t -= T
        state = isa(next_state, Function) ? next_state() : next_state
        T, next_state = random_transition(model, state)
    end
    observation(model, state), HiddenState(model, state, T-t, next_state)
end

"""
```julia
os, new_hs = next_observation(hs, ts)
```

Call `next_observation` repeatedly with the given times `ts`, collect the resulting
observations, and return them and the new hidden state.
"""
function next_observations(hs::HiddenState, ts)
    observations = [((o, hs) = next_observation(hs, t); o) for t in ts]
    observations, hs
end

end # module
