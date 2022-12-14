#=
This script examines the computational acceleration of the propsoed approach for different sized systems.
Details about each function can be found in ctesn_functions.jl or by "?" followed by the function name in REPL 
=#
using Pkg
Pkg.activate(".")
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
Random.seed!(1234)


file_dir = joinpath(pwd(), "src",)
include(joinpath(file_dir, "models/system_models.jl"))
include(joinpath(file_dir, "ctesn_functions.jl"))
include(joinpath(file_dir, "experimentParameters.jl")) # This is where all the experimental variables are defined

psid_times = []
ctesn_times = []
train_times = []

testParams = QuasiMonteCarlo.sample(acc_test_size, LB, UB, QuasiMonteCarlo.SobolSample()) # Sample parameter sapce to generate test samples

for sysSize in sysSizes

    sys = build_system(sysSize)

    busCap, totalGen, ibrBus, ibrGen, syncGen = getSystemProperties(sys);

    tripGen = [gen for gen in syncGen if occursin("Trip", gen.name)][1]
    genTrip = GeneratorTrip(tripTime, PSY.get_component(PSY.DynamicGenerator, sys, tripGen.name))

    rSol, N, stateIndex, stateLabels, simStep, resSize, res_time = simulate_reservoir(sys, maxSimStep, genTrip, tripGen.name);
    rSol = reduce(hcat, rSol.(simStep))

    total_train_time = @elapsed trainParams, rbfWeights, surr, psid_time = nonlinear_mapping!(sys, busCap, ibrBus, LB, UB, trainSamples, totalGen, rSol, stateIndex, simStep, genTrip, true); 
    D = SURR._construct_rbf_interp_matrix(surr.x, first(surr.x), surr.lb, surr.ub, surr.phi, surr.dim_poly, surr.scale_factor, surr.sparse)

    beta_rbf_train_time = @elapsed betaSurr = RadialBasis(trainParams, rbfWeights, LB, UB, rad = cubicRadial) # Fit an RBF to map from parameters to RBF weights

    psid_time=reduce(hcat, psid_time)

    numSteps = length(simStep)
    ctesn_time = @elapsed predict=nonlinear_predict.(eachcol(testParams), Ref(surr), Ref(betaSurr), Ref(D), Ref(numSteps))

    train_time =  res_time + (total_train_time-sum(psid_time[3, :])) + beta_rbf_train_time

    push!(psid_times, psid_time)
    push!(ctesn_times, ctesn_time)
    push!(train_times, train_time)

end

psid_mean = [mean(run[3, :]) for run in psid_times]
ctesn_mean = ctesn_times ./acc_test_size
acc = psid_mean ./ ctesn_mean

CSV.write("results/data/timingBenchmarkResults.csv", DataFrame(hcat(sysSizes, train_times, psid_mean, ctesn_mean, acc), :auto), header = ["SystemSize", "Train Time", "PSID", "CTESN", "Acceleration"])
