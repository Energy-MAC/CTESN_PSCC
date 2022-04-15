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
include(joinpath(file_dir, "experimentParameters.jl"))

psid_times = []
ctesn_times = []

testParams = QuasiMonteCarlo.sample(acc_test_size, LB, UB, QuasiMonteCarlo.SobolSample()) # Sample parameter sapce to generate test samples

for sysSize in sysSizes

    sys = build_system(sysSize)

    busCap, totalGen, ibrBus, ibrGen, syncGen = getSystemProperties(sys);

    tripGen = [gen for gen in syncGen if occursin("Trip", gen.name)][1]
    genTrip = GeneratorTrip(tripTime, PSY.get_component(PSY.DynamicGenerator, sys, tripGen.name))

    rSol, N, stateIndex, simStep, resSize = simulate_reservoir(sys, maxSimStep, genTrip, tripGen.name);

    trainParams, rbfWeights, surr = nonlinear_mapping!(sys, busCap, ibrBus, LB, UB, trainSamples, totalGen, rSol, stateIndex, simStep, genTrip); # Get RBF weights, trainParams, that map r(t) to x(t)
    D = SURR._construct_rbf_interp_matrix(surr.x, first(surr.x), surr.lb, surr.ub, surr.phi, surr.dim_poly, surr.scale_factor, surr.sparse)

    betaSurr = RadialBasis(trainParams, rbfWeights, LB, UB, rad = cubicRadial) # Build RBF that maps parmaters, p, to trainParams 

    psid_time = simSystem!.(Ref(sys), eachcol(testParams), Ref(ibrBus), Ref(busCap), Ref(totalGen), Ref(simStep), Ref(genTrip), Ref(true))
    psid_time=reduce(hcat, psid_time)

    numSteps = length(simStep)
    ctesn_time = @elapsed predict=nonlinear_predict.(eachcol(testParams), Ref(surr), Ref(betaSurr), Ref(D), Ref(numSteps))

    push!(psid_times, psid_time)
    push!(ctesn_times, ctesn_time)

end

psid_mean = [mean(run[3, :]) for run in psid_times]
ctesn_mean = ctesn_times/acc_test_size
acc = psid_mean ./ ctesn_mean

CSV.write("results/data/timingBenchmarkResults.csv", DataFrame(hcat(sysSizes, psid_mean, ctesn_mean, acc), :auto), header = ["SystemSize", "PSID", "CTESN", "Acceleration"])
