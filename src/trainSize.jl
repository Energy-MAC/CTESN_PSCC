using Pkg
Pkg.activate(".")
ENV["GKSwstype"] = "100"
using PowerSimulationsDynamics
PSID = PowerSimulationsDynamics
using PowerSystems
using Logging
configure_logging(console_level = Logging.Error)
using Sundials
using LightGraphs
using Plots
using OrdinaryDiffEq
using QuasiMonteCarlo
using LinearAlgebra
using Surrogates
SURR = Surrogates
using CSV
using DataFrames
using Statistics
using Distributed
using Random
using Plots.PlotMeasures
using Interpolations
using StatsPlots
Sp = StatsPlots


file_dir = joinpath(pwd(), "src",)
include(joinpath(file_dir, "models/system_models.jl"))
include(joinpath(file_dir, "ctesn_functions.jl"))
include(joinpath(file_dir, "experimentParameters.jl")) # This is where all the experimental variables are defined

sys = build_14bus()   # Build the system  
busCap, totalGen, ibrBus, ibrGen, syncGen = getSystemProperties(sys);

gen = PSY.get_component(PSY.DynamicGenerator, sys, "generator-2-Trip") # Get the generator we want to disconnect (Not-ML)
genTrip = GeneratorTrip(tripTime, gen) # Build perturbation to trip generator (Not-ML)

rSol, N, stateIndex, stateLabels, simStep, resSize = simulate_reservoir(sys, maxSimStep, genTrip, gen.name); # Simulate system and use solution to drive reservoir
rSol = reduce(hcat, rSol.(simStep))
numSteps = length(simStep)
freqIndex= 1:length(syncGen)-1

testParams = QuasiMonteCarlo.sample(testSize, LB, UB, QuasiMonteCarlo.SobolSample()) # Sample parameter sapce to generate test samples

testNadirErrors=[]
testFreqErrors=[]
testRocofErrors=[]

actual=simSystem!.(Ref(sys), eachcol(testParams), Ref(ibrBus), Ref(busCap), Ref(totalGen), Ref(interpolateTime), Ref(genTrip))

for i in 1:length(trainSizes) # Loop trough the different train sizes
    trainParams, Wouts, surr = nonlinear_mapping!(sys, busCap, ibrBus, LB, UB, trainSizes[i], totalGen, rSol, stateIndex, simStep, genTrip); # Get RBF weights, trainParams, that map r(t) to x(t)
    D = SURR._construct_rbf_interp_matrix(surr.x, first(surr.x), surr.lb, surr.ub, surr.phi, surr.dim_poly, surr.scale_factor, surr.sparse)
    betaSurr = RadialBasis(trainParams, Wouts, LB, UB, rad = cubicRadial) # Build RBF that maps parmaters, p, to trainParams 

    predict=nonlinear_predict.(eachcol(testParams), Ref(surr), Ref(betaSurr), Ref(D), Ref(numSteps))
    predict=[transpose(p) for p in predict]        
    resamplePred=resmaplePrediction(predict, simStep, interpolateTime)
            
    nadir_errors = [minimum(actual[i][stateIndex[freqIndex], :]) - minimum(resamplePred[i][freqIndex, :]) for i in 1:testSize]
    max_errors = [norm(actual[i][stateIndex[freqIndex], :] - resamplePred[i][freqIndex, :], Inf) for i in 1:testSize]
    rocof_errors = [(minimum(diff(actual[i][stateIndex[freqIndex], :], dims=2))-minimum(diff(resamplePred[i][freqIndex, :], dims=2)))/interpolateStep for i in 1:testSize]

    push!(testNadirErrors, nadir_errors)
    push!(testFreqErrors, max_errors)
    push!(testRocofErrors, rocof_errors)
end

testNadirErrors=reduce(hcat, testNadirErrors)
testFreqErrors=reduce(hcat, testFreqErrors)
testRocofErrors=reduce(hcat, testRocofErrors);

NadirErrormHz = 1e3*freq_base.*testNadirErrors
FreqErrormHz = 1e3*freq_base.*testFreqErrors
RoCoFErrormHz = 1e3*freq_base.*testRocofErrors

CSV.write("results/data/trainSizeFreqErrors.csv", DataFrame(FreqErrormHz, :auto), header = string.(trainSizes))
CSV.write("results/data/trainSizeRocofErrors.csv", DataFrame(RoCoFErrormHz, :auto), header = string.(trainSizes))
CSV.write("results/data/trainSizeNadirErrors.csv", DataFrame(NadirErrormHz, :auto), header = string.(trainSizes))

Sp.boxplot(NadirErrormHz, label="", xlabel="Training Size", ylabel="Nadir prediction error [mHz]")
Sp.xticks!([1:1:length(trainSizes);], string.(trainSizes))
savefig("results/figs/boxplotNadirError.pdf")

boxplot(FreqErrormHz, label="", xlabel="Training Size", ylabel="Maximum prediction error [mHz]")
xticks!([1:1:length(trainSizes);], string.(trainSizes))
savefig("results/figs/boxplotMaxError.pdf")

boxplot(RoCoFErrormHz, label="", xlabel="Training Size", ylabel="Maximum RoCoF error [mHz/s]")
xticks!([1:1:length(trainSizes);], string.(trainSizes))
savefig("results/figs/boxplotRoCoFError.pdf")