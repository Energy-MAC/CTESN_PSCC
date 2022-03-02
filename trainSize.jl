using Pkg
Pkg.activate(".")
ENV["GKSwstype"] = "100"
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
using Interpolations
using StatsPlots # no need for `using Plots` as that is reexported here
Random.seed!(1234)

file_dir = joinpath(pwd(), "src",)
include(joinpath(file_dir, "models/system_models.jl"))
include(joinpath(file_dir, "ctesn_functions.jl"))
include(joinpath(file_dir, "experimentParameters.jl"))

sys, busCap, totalPower = buid_system(ibrBus, GF, Gf);   # Build the system
gen = PSY.get_component(PSY.DynamicGenerator, sys, "generator-2-Trip")
genTrip = GeneratorTrip(tripTime, gen)

rSol, N, stateIndex, simStep = simulate_reservoir(sys, resSize, maxSimStep);

testParams = QuasiMonteCarlo.sample(testSize, LB, UB, QuasiMonteCarlo.SobolSample()) # Sample parameter sapce

freq_error=zeros(testSize, length(trainSizes))
rocof_error=zeros(testSize,length(trainSizes))
nadir_error=zeros(testSize,length(trainSizes))

for i in 1:length(trainSizes)
    
    trainParams, Wouts = linear_mapping(sys, busCap, LB, UB, trainSizes[i], totalPower, rSol, stateIndex, simStep);
    surr = RadialBasis(trainParams, Wouts, LB, UB, rad = linearRadial)
    
    for j in 1:testSize
        Gf=testParams[1,j]*(1-testParams[2,j]) # Grid following %
        GF=testParams[1,j]*testParams[2,j] # Grid forming %
        
        global sys=change_ibr_penetration(sys, GF, Gf, ibrBus, busCap, totalPower) # Change generation mix at each of the IBR nodes
        sim = Simulation!(
            ResidualModel, #Type of model used
            sys,         #system
            file_dir,       #path for the simulation output
            tspan, #time span
            genTrip,
        )

        res = small_signal_analysis(sim);

        execute!(sim, IDA())
   
        sol_array = Array(sim.results.solution(interpolateTime)) # Convert physcial solution to an array
        pred = linear_predict([testParams[1,j], testParams[2,j]], surr, rSol, simStep, N, resSize)
        resamplePred=transpose(reduce(hcat, [LinearInterpolation(simStep, pred[i, :]).(interpolateTime) for i in 1:size(pred)[1]]))
                
        nadir_error[j,i]=minimum(sol_array[stateIndex, :]) - minimum(pred) # Find difference between lowest predicted frequency and actual lowest frequency
        freq_error[j,i]=norm(sol_array[stateIndex, :] - resamplePred, Inf) # Find the largest error
        rocof_error[j,i]=(minimum(diff(sol_array[stateIndex, :], dims=2))-minimum(diff(resamplePred, dims=2)))/interpolateStep # Find the largest error of numerically 
    end
end

nadir_plot=boxplot(freq_base*nadir_error[:,1], label="", xlabel="Training Size", ylabel="Nadir prediction error [Hz]")
for i in 2:length(trainSizes)
    boxplot!(freq_base*nadir_error[:, i], label="")
end
xticks!([1:1:length(trainSizes);], string.(trainSizes))
savefig("results/figs/trasinSizeNadirError.pdf")

max_error_plot=boxplot(freq_base*freq_error[:,1], label="", xlabel="Training Size", ylabel="Maximum prediction error [Hz]")
for i in 2:length(trainSizes)
    boxplot!(freq_base*freq_error[:,i], label="")
end
xticks!([1:1:length(trainSizes);], string.(trainSizes))
savefig("results/figs/trasinSizeMaxError.pdf")

rocof_plot=boxplot(freq_base*rocof_error[:,1], label="", xlabel="Training Size", ylabel="Maximum RoCoF error [Hz/s]")
for i in 2:length(trainSizes)
    boxplot!(freq_base*rocof_error[:,i], label="")
end
xticks!([1:1:length(trainSizes);], string.(trainSizes))
savefig("results/figs/trasinSizeMaxError.pdf")

CSV.write("results/data/freq_test_errors.csv", DataFrame(freq_error, :auto), header = string.(trainSizes))
CSV.write("results/data/rocof_test_errors.csv", DataFrame(rocof_error, :auto), header = string.(trainSizes))
CSV.write("results/data/nadir_test_errors.csv", DataFrame(nadir_error, :auto), header = string.(trainSizes))