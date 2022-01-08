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
using Interpolations
Random.seed!(1234)

file_dir = joinpath(pwd(), "src",)
include(joinpath(file_dir, "models/system_models.jl"))
include(joinpath(file_dir, "ctesn_functions.jl"))
include(joinpath(file_dir, "experimentParameters.jl"))


sys, busCap, totalPower = buid_system(ibrBus, GF, Gf);   # Build the system

rSol, N, stateIndex, simStep = simulate_reservoir(sys, resSize, maxSimStep);

trainParams, Wouts = linear_mapping(sys, busCap, LB, UB, trainSamples, totalPower, rSol, stateIndex, simStep);

surr = RadialBasis(trainParams, Wouts, LB, UB, rad = linearRadial)

Inv = [LB[1]:paramSweepStep:UB[1];]
GF = [LB[2]:paramSweepStep:UB[2];]
nadir=zeros(length(GF),length(Inv));
rocof=zeros(length(GF),length(Inv));
settling_time=zeros(length(GF),length(Inv));

for i in 1:length(GF)
    for j in 1:length(Inv) 

        pred = linear_predict([Inv[j], GF[i]], surr, rSol, simStep, N, resSize)
        resamplePred=transpose(reduce(hcat, [LinearInterpolation(simStep, pred[i, :]).(interpolateTime) for i in 1:size(pred)[1]]))
        reverse_freq = vec(reverse(mean(resamplePred, dims=1)))
        top_margin = reverse_freq[1]+settlingBand
        bottom_margin = reverse_freq[1]-settlingBand
        
        settling_time[i,j]=(tStop-tripTime)-findfirst((reverse_freq .< bottom_margin) .| (reverse_freq .> top_margin))*interpolateStep
        nadir[i,j]=minimum(pred)
        rocof[i,j]=minimum(diff(resamplePred, dims=2))/interpolateStep
    end
end

heatmap(100*Inv, 100*GF, freq_base*nadir, xlabel="% CIG", ylabel="% of CIG that are GF", colorbar_title=" \nFrequency Nadir [Hz]", margin=4mm, c=:viridis)
savefig("results/figs/nadir_heatmap.pdf")

heatmap(100*Inv, 100*GF, freq_base*rocof, xlabel="% CIG", ylabel="% of CIG that are GF", colorbar_title=" \nRoCoF [Hz/s]", margin=3mm, c=:viridis)
savefig("results/figs/rocof_heatmap.pdf")

heatmap(100*Inv, 100*GF, settling_time, xlabel="% CIG", ylabel="% of CIG that are GF", colorbar_title=" \nSettling Time [s]", margin=3mm, c=:viridis)
savefig("results/figs/settlingTime_heatmap.pdf")