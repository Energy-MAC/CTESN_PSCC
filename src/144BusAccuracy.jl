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

file_dir = joinpath(pwd(), "src",)
include(joinpath(file_dir, "models/system_models.jl"))
include(joinpath(file_dir, "ctesn_functions.jl"))
include(joinpath(file_dir, "experimentParameters.jl")) # This is where all the experimental variables are defined

sysSize = 144
testSize= 50
trainSamples = 25
sys = build_system(sysSize)

busCap, totalGen, ibrBus, ibrGen, syncGen = getSystemProperties(sys);

gen = [gen for gen in syncGen if occursin("Trip", gen.name)][1]
genTrip = GeneratorTrip(tripTime, PSY.get_component(PSY.DynamicGenerator, sys, gen.name))

rSol, N, stateIndex, stateLabels, simStep, resSize, res_time = simulate_reservoir(sys, maxSimStep, genTrip, gen.name); # Simulate system and use solution to drive reservoir
rSol = reduce(hcat, rSol.(simStep))
numSteps = length(simStep);
freqIndex= 1:length(syncGen)-1

testParams = QuasiMonteCarlo.sample(testSize, LB, UB, QuasiMonteCarlo.SobolSample()) # Sample parameter sapce to generate test samples

actual=simSystem!.(Ref(sys), eachcol(testParams), Ref(ibrBus), Ref(busCap), Ref(totalGen), Ref(interpolateTime), Ref(genTrip))

trainParams, Wouts, surr = nonlinear_mapping!(sys, busCap, ibrBus, LB, UB, trainSamples, totalGen, rSol, stateIndex, simStep, genTrip); # Get RBF weights, trainParams, that map r(t) to x(t)
D = SURR._construct_rbf_interp_matrix(surr.x, first(surr.x), surr.lb, surr.ub, surr.phi, surr.dim_poly, surr.scale_factor, surr.sparse)
betaSurr = RadialBasis(trainParams, Wouts, LB, UB, rad = cubicRadial) # Build RBF that maps parmaters, p, to trainParams 

predict=nonlinear_predict.(eachcol(testParams), Ref(surr), Ref(betaSurr), Ref(D), Ref(numSteps))
predict=[transpose(p) for p in predict]        
resamplePred=resmaplePrediction(predict, simStep, interpolateTime)
            
Vr_index=findall( x -> occursin("V", x) && occursin("_R", x), stateLabels)
Vi_index=findall( x -> occursin("V", x) && occursin("_I", x), stateLabels)
Ir_index=findall( x -> occursin("Bus", x) && occursin("_R", x), stateLabels)
Ii_index=findall( x -> occursin("Bus", x) && occursin("_I", x), stateLabels)

v_mag_psid = [sqrt.(actual[i][stateIndex[Vr_index], :].^2+actual[i][stateIndex[Vi_index], :].^2) for i in 1:testSize]
v_mag_ctesn = [sqrt.(resamplePred[i][Vr_index, :].^2+resamplePred[i][Vi_index, :].^2) for i in 1:testSize]

i_mag_psid = [sqrt.(actual[i][stateIndex[Ir_index], :].^2+actual[i][stateIndex[Ii_index], :].^2) for i in 1:testSize]
i_mag_ctesn = [sqrt.(resamplePred[i][Ir_index, :].^2+resamplePred[i][Ii_index, :].^2) for i in 1:testSize]

freq_rmse = [sqrt.(sum((actual[i][stateIndex[freqIndex], :] - resamplePred[i][freqIndex, :]) .^ 2, dims=2) ./ size(predict[1])[2]) for i in 1:testSize]
v_rmse = [sqrt.(sum((v_mag_psid[i] - v_mag_ctesn[i]) .^ 2, dims=2) ./ size(predict[1])[2]) for i in 1:testSize]
i_rmse = [sqrt.(sum((i_mag_psid[i] - i_mag_ctesn[i]) .^ 2, dims=2) ./ size(predict[1])[2]) for i in 1:testSize]

freq_rmse=reduce(vcat, freq_rmse)
v_rmse=reduce(vcat, v_rmse)
i_rmse=reduce(vcat, i_rmse);

CSV.write("results/data/144CtesnFreqErrors.csv", DataFrame(freq_rmse, :auto), header = ["Frequency"])
CSV.write("results/data/144CtesnVoltageErrors.csv", DataFrame( v_rmse, :auto), header = ["Voltage"])
CSV.write("results/data/144CtesnCurrentErrors.csv", DataFrame(i_rmse, :auto), header = ["Current"])
