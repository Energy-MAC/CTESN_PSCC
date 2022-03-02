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
include(joinpath(file_dir, "experimentParameters.jl")) # This is where all the experimental variables are defined

global sys, busCap, totalPower = buid_system(ibrBus, GF, Gf);   # Build the system (Not-ML)
gen = PSY.get_component(PSY.DynamicGenerator, sys, "generator-2-Trip") # Get the generator we want to disconnect (Not-ML)
genTrip = GeneratorTrip(tripTime, gen) # Build perturbation to trip generator (Not-ML)

rSol, N, stateIndex, simStep = simulate_reservoir(sys, resSize, maxSimStep); # Simulate system and use solution to drive reservoir

testParams = QuasiMonteCarlo.sample(testSize, LB, UB, QuasiMonteCarlo.SobolSample()) # Sample parameter sapce to generate test samples

freq_error=zeros(testSize, length(trainSizes)) # Initilaize matrics to store resutls 
rocof_error=zeros(testSize,length(trainSizes)) # Initilaize matrics to store resutls 
nadir_error=zeros(testSize,length(trainSizes)) # Initilaize matrics to store resutls 

for i in 1:length(trainSizes) # Loop trough the different test sizes
    
    trainParams, Wouts, surr = nonlinear_mapping(sys, busCap, LB, UB, trainSizes[i], totalPower, rSol, stateIndex, simStep); # Get RBF weights, trainParams, that map r(t) to x(t)
    betaSurr = RadialBasis(trainParams, Wouts, LB, UB, rad = cubicRadial) # Build RBF that maps parmaters, p, to trainParams 
    
    for j in 1:testSize
        Gf=testParams[1,j]*(1-testParams[2,j]) # Grid following % (Not-ML)
        GF=testParams[1,j]*testParams[2,j] # Grid forming % (Not-ML)
        
        global sys=change_ibr_penetration(sys, GF, Gf, ibrBus, busCap, totalPower) # Change generation mix at each of the IBR nodes (Not-ML)
        sim = Simulation!(
            ResidualModel, #Type of model used
            sys,         #system
            file_dir,       #path for the simulation output
            tspan, #time span
            genTrip,
        )

        res = small_signal_analysis(sim); # Perform small-signal analysis to ensure pysical system is stalbe (Not-ML)

        execute!(sim, IDA()) # Simualte system (Not-ML)
   
        sol_array = Array(sim.results.solution(interpolateTime)) # Interpoalte solution of system to desired time step
        pred=nonlinear_predict(testParams[:,j], betaSurr, surr, rSol, simStep, N, resSize) # Use CTESN to predict solution of the system
        resamplePred=transpose(reduce(hcat, [LinearInterpolation(simStep, pred[i, :]).(interpolateTime) for i in 1:size(pred)[1]])) # Re-sample prediciton by linear interpolation 
                
        nadir_error[j,i]=minimum(sol_array[stateIndex, :]) - minimum(pred) # Find difference between lowest predicted frequency and actual lowest frequency (Not-ML)
        freq_error[j,i]=norm(sol_array[stateIndex, :] - resamplePred, Inf) # Find the largest error (Not-ML)
        rocof_error[j,i]=(minimum(diff(sol_array[stateIndex, :], dims=2))-minimum(diff(resamplePred, dims=2)))/interpolateStep # Find the largest error of numerically (Not-ML)
    end
end

## The remainder of the code just creates plots and saves the results

nadir_plot=boxplot(freq_base*nadir_error[:,1], label="", xlabel="Training Size", ylabel="Nadir prediction error [Hz]")
for i in 2:length(trainSizes)
    boxplot!(freq_base*nadir_error[:, i], label="")
end
xticks!([1:1:length(trainSizes);], string.(trainSizes))
savefig("results/figs/nonLinearNadirError.pdf")

max_error_plot=boxplot(freq_base*freq_error[:,1], label="", xlabel="Training Size", ylabel="Maximum prediction error [Hz]")
for i in 2:length(trainSizes)
    boxplot!(freq_base*freq_error[:,i], label="")
end
xticks!([1:1:length(trainSizes);], string.(trainSizes))
savefig("results/figs/nonLinearMaxError.pdf")

rocof_plot=boxplot(freq_base*rocof_error[:,1], label="", xlabel="Training Size", ylabel="Maximum RoCoF error [Hz/s]")
for i in 2:length(trainSizes)
    boxplot!(freq_base*rocof_error[:,i], label="")
end
xticks!([1:1:length(trainSizes);], string.(trainSizes))
savefig("results/figs/nonLinearRocofError.pdf")

CSV.write("results/data/nonLinearFreqErrors.csv", DataFrame(freq_error, :auto), header = string.(trainSizes))
CSV.write("results/data/nonLinearRocofErrors.csv", DataFrame(rocof_error, :auto), header = string.(trainSizes))
CSV.write("results/data/nonLinearNadirErrors.csv", DataFrame(nadir_error, :auto), header = string.(trainSizes))