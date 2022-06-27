using Pkg
Pkg.activate(".")
using PowerSimulationsDynamics
PSID = PowerSimulationsDynamics
using PowerSystems
using Logging
using Sundials
using LightGraphs
using OrdinaryDiffEq
using QuasiMonteCarlo
using LinearAlgebra
using Surrogates
using CSV
using DataFrames
using Statistics
using Distributed
using Random
using Interpolations
using CairoMakie
using DelimitedFiles
SURR = Surrogates

Random.seed!(1234)


file_dir = joinpath(pwd(), "src",)
include(joinpath(file_dir, "models/system_models.jl"))
include(joinpath(file_dir, "ctesn_functions.jl"))
include(joinpath(file_dir, "experimentParameters.jl"))


sys = build_14bus()   # Build the system

busCap, totalGen, ibrBus, ibrGen, syncGen = getSystemProperties(sys);
freqIndex= 1:length(syncGen)-1

tripGen = [gen for gen in syncGen if occursin("Trip", gen.name)][1]
genTrip = GeneratorTrip(tripTime, PSY.get_component(PSY.DynamicGenerator, sys, tripGen.name));

rSol, N, stateIndex, stateLabels, simStep, resSize = simulate_reservoir(sys, maxSimStep, genTrip, tripGen.name);
rSol = reduce(hcat, rSol.(simStep))

trainParams, rbfWeights, surr = nonlinear_mapping!(sys, busCap, ibrBus, LB, UB, trainSamples, totalGen, rSol, stateIndex, simStep, genTrip); # Get RBF weights, trainParams, that map r(t) to x(t)
betaSurr = RadialBasis(trainParams, rbfWeights, LB, UB, rad = cubicRadial) # Build RBF that maps parmaters, p, to trainParams 

D = SURR._construct_rbf_interp_matrix(surr.x, first(surr.x), surr.lb, surr.ub, surr.phi, surr.dim_poly, surr.scale_factor, surr.sparse);

Inv = [LB[1]:paramSweepStep:UB[1];]
GF = [LB[2]:paramSweepStep:UB[2];]

paramPairs=vec(collect(Iterators.product(Inv, GF)))
paramMatrix=hcat((collect(row) for row in paramPairs)...)

numSteps = length(simStep)
predict=nonlinear_predict.(eachcol(paramMatrix), Ref(surr), Ref(betaSurr), Ref(D), Ref(numSteps));
predict=[transpose(p) for p in predict]

resamplePred=resmaplePrediction(predict, simStep, interpolateTime)
meanFrequency = [mean(resamplePred[i][freqIndex, :], dims=1) for i in 1:size(predict)[1]]

nadir = [minimum(predict[i][freqIndex, :]) for i in 1:size(predict)[1]]
rocof = [minimum(diff(resamplePred[i][freqIndex, :], dims=2))/interpolateStep for i in 1:size(predict)[1]]

settling_time = [getSettlingTime(resamplePred[i][freqIndex, :]) for i in 1:size(predict)[1]]

fig=Figure()
ax, hm = heatmap(fig[1, 1], 100*paramMatrix[1,:], 100*paramMatrix[2,:], freq_base*nadir)
Colorbar(fig[1, 2], hm, label="Frequency Nadir [Hz]")
ax.ylabel = "% of CIG that are GF"
ax.xlabel = "% CIG"
fig 
#savefig("results/figs/nadir_heatmap.pdf")

fig=Figure()
ax, hm = heatmap(fig[1, 1], 100*paramMatrix[1,:], 100*paramMatrix[2,:], freq_base*rocof)
Colorbar(fig[1, 2], hm, label="RoCoF [Hz/s]")
ax.ylabel = "% of CIG that are GF"
ax.xlabel = "% CIG"
fig 
#savefig("results/figs/rocof_heatmap.pdf")

fig=Figure()
ax, hm = heatmap(fig[1, 1], 100*paramMatrix[1,:], 100*paramMatrix[2,:], settling_time)
Colorbar(fig[1, 2], hm, label="Settling Time [s]")
ax.ylabel = "% of CIG that are GF"
ax.xlabel = "% CIG"
fig 
#savefig("results/figs/settlingTime_heatmap.pdf")


open("results/data/heatmap.dat", "w") do io
    for index in eachindex(paramMatrix[1, :])
        if index % length(Inv) == 1
            writedlm(io," ")
        end
        writedlm(io, [100*paramMatrix[2, index] 100*paramMatrix[1, index]  round(freq_base*nadir[index], digits=5) round(freq_base*rocof[index], digits=5) round(settling_time[index], digits=5)])
    end 
end
