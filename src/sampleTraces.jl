#using Pkg
#Pkg.activate(".")
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
include(joinpath(file_dir, "experimentParameters.jl"))

sys = build_14bus()   # Build the system

busCap, totalGen, ibrBus, ibrGen, syncGen = getSystemProperties(sys);

tripGen = [gen for gen in syncGen if occursin("Trip", gen.name)][1]
genTrip = GeneratorTrip(tripTime, PSY.get_component(PSY.DynamicGenerator, sys, tripGen.name));

rSol, N, stateIndex, stateLabels, simStep, resSize  = simulate_reservoir(sys, maxSimStep, genTrip, tripGen.name);
rSol = reduce(hcat, rSol.(simStep))

trainParams, rbfWeights, surr = nonlinear_mapping!(sys, busCap, ibrBus, LB, UB, trainSamples, totalGen, rSol, stateIndex, simStep, genTrip); # Get RBF weights, trainParams, that map r(t) to x(t)
betaSurr = RadialBasis(trainParams, rbfWeights, LB, UB, rad = cubicRadial) # Build RBF that maps parmaters, p, to trainParams 

testParams=[0.6 0.34; 0.6 0.17; 0.2 0.25] # Sample test points used to compare predicted to true solution in sampleTraces.jl
D = SURR._construct_rbf_interp_matrix(surr.x, first(surr.x), surr.lb, surr.ub, surr.phi, surr.dim_poly, surr.scale_factor, surr.sparse);

numSteps = length(simStep)
predict=nonlinear_predict.(eachrow(testParams), Ref(surr), Ref(betaSurr), Ref(D), Ref(numSteps));
predict=[transpose(p) for p in predict]

actual = simSystem!.(Ref(sys), eachrow(testParams), Ref(ibrBus), Ref(busCap), Ref(totalGen), Ref(simStep), Ref(genTrip))

p1 = plot(simStep, freq_base*predict[1][1, :], label="CIG=60% GF=20% Gf=40%", linewidth=3, xlabel="Time [s]", ylabel="Frequency [Hz]", legend=:bottomright)
plot!(simStep, freq_base*predict[2][1, :], label="CIG=60% GF=10% Gf=50%", linewidth=3)
plot!(simStep, freq_base*predict[3][1, :], label="CIG=20% GF=5% Gf=15%", linewidth=3)
for i = 1:3
    plot!(simStep, freq_base*actual[i][stateIndex[1], :], color="black", label="", linewidth=1.5, linestyle=:dash)
end 
p2 = plot(simStep, 1e3*freq_base*(predict[1][1, :]-actual[1][stateIndex[1], :]), label="", linewidth=2, xlabel="Time [s]", ylabel="Prediction Error [mHz]", legend=:bottomright)
for i = 2:3 
    plot!(simStep, 1e3*freq_base*(predict[i][1, :]-actual[i][stateIndex[1], :]), label="", linewidth=2)
end
plot(p1, p2, layout=(2,1), margin=3mm)
plot!(size=(400,600))
savefig("results/figs/sample_traces_subplots.pdf")

resamplePred=resmaplePrediction(predict, simStep, interpolateTime)
resampleSol=resmaplePrediction(actual, simStep, interpolateTime)

p1=plot(interpolateTime, freq_base*resamplePred[1][1, :], label="CIG=60% GF=20% Gf=40%", linewidth=3, xlabel="Time [s]", ylabel="Frequency [Hz]", legend=:bottomright)
plot!(interpolateTime, freq_base*resamplePred[2][1, :], label="CIG=60% GF=10% Gf=50%", linewidth=3)
plot!(interpolateTime, freq_base*resamplePred[3][1, :], label="CIG=20% GF=5% Gf=15%", linewidth=3)



for i in 1:3
    CSV.write(string("results/data/sampleTraces/Case_", Int(100*testParams[i,1]), "_IBR_", Int(100*testParams[i,2]), "_GF.csv"), DataFrame(hcat(Array(interpolateTime), round.(freq_base .*resamplePred[i][1, :], digits=5),  round.(freq_base .*resampleSol[i][stateIndex[1], :], digits=5)), :auto), header = ["times", "CTESN", "PSID"])
end

for i in 1:3
    CSV.write(string("results/data/sampleTraces/CTESN_", testParams[i,1], "_IBR_", testParams[i,2], "_GF.csv"), DataFrame(hcat(Array(interpolateTime), transpose(resamplePred[i])), :auto), header = vcat(["times"], stateLabels))
    CSV.write(string("results/data/sampleTraces/PSID_", testParams[i,1], "_IBR_", testParams[i,2], "_GF.csv"), DataFrame(hcat(Array(interpolateTime), transpose(resampleSol[i][stateIndex, :])), :auto), header = vcat(["times"], stateLabels))
end