using HiddenPoisson
using Parameters
using Base.Test

import HiddenPoisson: rates, next_state, observation

"Count the (positive) integers in `itr`, returning a vector of counts."
function count_integers(itr, largest = maximum(itr))
    counts = zeros(Int, largest)
    for i in itr
        counts[i] += 1
    end
    counts
end

function test_competing_poisson(rates, N=10000)
    Tis = [HiddenPoisson.competing_poisson(rates) for _ in 1:N]
    @test isapprox(mean(first.(Tis)), 1/sum(rates); rtol = 0.05)
    @test isapprox(normalize(count_integers((last(Ti) for Ti in Tis), length(rates)), 1),
                   normalize(rates, 1); rtol = 0.1)
end

@testset "competing poisson" begin
    for _ in 1:1000
        N = rand(2:10)
        rates = abs.(randn(10))
        test_competing_poisson(rates)
    end
end

"Diamond-Mortensen-Pissarides model in continuous time.

# States

1. employment
2. unemployment

# Shocks

employment: job finding (at rate λ)
unemployment: separation (at rate σ)

# Observations

:E and :U according to the state (deterministic).
"
@with_kw immutable DMPmodel{T}
    λ::T                        # job finding rate
    σ::T                        # separation rate
end

rates(m::DMPmodel, state) = state == 1 ? [m.σ] : [m.λ]

next_state(m::DMPmodel, state, i) = 3-state

observation(m::DMPmodel, state) = [:E, :U][state]

employment_rate(m::DMPmodel) = model.λ/(model.λ + model.σ)

model = DMPmodel(λ = 0.4, σ = 0.1)

@testset "hidden state steps" begin
    hs1 = HiddenState(model, 1, 2.5, 2)
    o1, hs2 = next_observation(hs1, 1)
    @test o1 == :E
    @test hs1.model ≡ hs2.model
    @test hs1.state ≡ hs2.state
    @test hs2.T == 1.5
    @test hs1.i == hs2.i
    o2, hs3 = next_observation(hs2, 1)
    @test o2 == :E
    @test hs1.model ≡ hs3.model
    @test hs1.state ≡ hs3.state
    @test hs3.T == 0.5
    @test hs1.i == hs3.i
end

@testset "hidden state constructor" begin
    @test isapprox(mean(HiddenState(model, 1).T for _ in 1:10000), 1/model.σ; rtol = 0.03)
end

@testset "hidden state simulation" begin
    hs = HiddenState(model, 1)
    count_E = 0
    N = 100000
    for _ in 1:N
        o, hs = next_observation(hs, 1/12)
        if o == :E
            count_E += 1
        end
    end
    @test isapprox(count_E/N, employment_rate(model); rtol = 0.03)
end

@testset "multiple observations" begin
    hs = HiddenState(model, 1)
    count_E = 0
    N = 0
    for _ in 1:10000
        ts = abs(randn(rand(1:10)))
        N += length(ts)
        os, hs = next_observations(hs, ts)
        @test length(os) == length(ts)
        for o in os
            if o == :E
                count_E += 1
            end
        end
    end
    @test isapprox(count_E/N, employment_rate(model); rtol = 0.03)
end
