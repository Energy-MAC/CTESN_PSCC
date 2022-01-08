using Pkg
Pkg.activate(".")
using PowerSimulationsDynamics
PSID = PowerSimulationsDynamics
using PowerSystems
using Logging
using Sundials
using LightGraphs
using Plots
using OrdinaryDiffEq
using QuasiMonteCarlo
using LinearAlgebra
using Surrogates
using CSV
using DataFrames
using Statistics
using Distributed
using Random
using Plots.PlotMeasures
Random.seed!(1234)

file_dir = joinpath(pwd(), "src",)
include(joinpath(file_dir, "models/system_models.jl"))
include(joinpath(file_dir, "ctesn_functions.jl"))
include(joinpath(file_dir, "experimentParameters.jl"))

sys, busCap, totalPower = buid_system(ibrBus, GF, Gf);   # Build the system
gen = PSY.get_component(PSY.DynamicGenerator, sys, "generator-2-Trip")
genTrip = GeneratorTrip(tripTime, gen)

rSol, N, stateIndex, simStep = simulate_reservoir(sys, resSize, maxSimStep);

trainParams, Wouts = linear_mapping(sys, busCap, LB, UB, trainSamples, totalPower, rSol, stateIndex, simStep);

surr = RadialBasis(trainParams, Wouts, LB, UB, rad = linearRadial)

predict = Matrix{Float64}[]
actual = Matrix{Float64}[]

for i in 1:length(testParams)
    Gf=testParams[i][1]*(1-testParams[i][2]) # Grid following %
    GF=testParams[i][1]*testParams[i][2] # Grid forming %

    sys=change_ibr_penetration(sys, GF, Gf, ibrBus, busCap, totalPower) # Change generation mix at each of the IBR nodes
    sim = Simulation!(
        ResidualModel, #Type of model used
        sys,         #system
        file_dir,       #path for the simulation output
        tspan, #time span
        genTrip,
    )

    res = small_signal_analysis(sim)

    execute!(sim, IDA())
        
    push!(actual, Array(sim.results.solution(simStep)))
    push!(predict, linear_predict(testParams[i], surr, rSol, simStep, N, resSize))
    
end 

p1 = plot(simStep, freq_base*predict[1][1, :], label="CIG=60% GF=30% Gf=30%", linewidth=3)
plot!(simStep, freq_base*predict[2][1, :], label="CIG=60% GF=10% Gf=50%", linewidth=3)
plot!(simStep, freq_base*predict[3][1, :], label="CIG=30% GF=5% Gf=25%", linewidth=3)
plot!(simStep, freq_base*actual[1][stateIndex[1], :], color="black", label="", linewidth=1.5, linestyle=:dash)
plot!(simStep, freq_base*actual[2][stateIndex[1], :], color="black", label="", linewidth=1.5, linestyle=:dash)
plot!(simStep, freq_base*actual[3][stateIndex[1], :], color="black", label="", linewidth=1.5, linestyle=:dash, xlabel="Time [s]", ylabel="Frequency [Hz]", legend=:bottomright,)

p2 = plot(simStep, freq_base*(predict[1][1, :]-actual[1][stateIndex[1], :]), label="", linewidth=2)
plot!(simStep, freq_base*(predict[2][1, :]-actual[2][stateIndex[1], :]), label="", linewidth=2)
plot!(simStep, freq_base*(predict[3][1, :]-actual[3][stateIndex[1], :]), label="", linewidth=2, xlabel="Time [s]", ylabel="Prediction Error [Hz]", legend=:bottomright)

plot(p1, p2, layout=(2,1), margin=3mm)
plot!(size=(400,600))
savefig("results/figs/sample_traces_subplots.pdf")