module HiddenPoisson

using ArgCheck
using Distributions
using Parameters
using StatsBase

export
    rates, next_state, observation, # users define methods for these
    HiddenState, next_observation, next_observations # simulation interface
    
"""
```julia
rates(model, state)
```

Return Poisson rates as a vector for the next event, given the `model` and the current
hidden state `state`.

The user needs to supply a method for a given `model`.
"""
function rates end

"""
```julia
next_state(model, state, i)
```

For a `model` in `state`, draw the next (possibly random) state, assuming that event of
index `i` arrived first.

The user needs to supply a method for a given `model`.
"""
function next_state end

"""
```julia
observation(model, state)

The user needs to supply a method for a given `model`.
```

Draw a (possibly random) observation of the `model` in `state`.
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
Representation of a hidden Poisson process.

`model` contains the model parameters, and is used by `rates` and `observation`.

`state` is the current state (an arbitrary object).

`T` is the time left until the next Poisson event, which is simulated using `rates`.

`i` is the index for the event which happens at `T`.
"""
immutable HiddenState{Tm, Ts, Tt <: Real}
    model::Tm
    state::Ts
    T::Tt
    i::Int
    function HiddenState(model, state, T, i)
        @argcheck T ≥ 0
        new(model, state, T, i)
    end
end

HiddenState{Tm,Ts,Tt}(model::Tm, state::Ts, T::Tt, i) =
    HiddenState{Tm,Ts,Tt}(model, state, T, i)

HiddenState(model, state) = HiddenState(model, state,
                                        competing_poisson(rates(model, state))...)

"""
```julia
o, new_hs = next_observation(hs, t)
```

Return observation `o` after time `t` starting from hidden state `hs`, and the new
hidden state.
"""
function next_observation(hs::HiddenState, t::Real)
    @argcheck t ≥ zero(t)
    @unpack model, state, T, i = hs
    while t ≥ T
        t -= T
        state = next_state(model, state, i)
        r = rates(model, state)
        T, i = competing_poisson(r)
    end
    observation(model, state), HiddenState(model, state, T-t, i)
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
