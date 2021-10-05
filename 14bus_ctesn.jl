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
using Distributed
using Surrogates
using CSV
using DataFrames
using Statistics
using SharedArrays
using Random
Random.seed!(1234)
gr()

function build_surrogate(sys, bus_cap, lb, ub, resSize, sample_points, total_power)
    
    sim = PSID.Simulation!(
        PSID.ImplicitModel, #Type of model used
        sys,         #system
        file_dir,       #path for the simulation output
        (0.0, 10.0), #time span
        console_level = Logging.Info,
    )

    res = small_signal_analysis(sim)
    gen = PSY.get_component(ThermalStandard, sys, "generator-2-Trip")
    PSY.set_available!(gen, false)
    
    global_state_index = PSID.get_global_index(sim.simulation_inputs);
    drop_idx=sort(collect(values(global_state_index["generator-2-Trip"])));
    x0_gen_trip = vcat(sim.x0_init[1:drop_idx[1]-1], sim.x0_init[drop_idx[end]+1:end]);

    sim_trip_gen = Simulation(
            ImplicitModel,
            sys,
            pwd(),
            (0.0,60.0);
            initialize_simulation = false,
            initial_conditions = x0_gen_trip,
            )
    sim_max_step=0:1:60
    execute!(sim_trip_gen, IDA(), tstops=sim_max_step)
    # execute!(sim_trip_gen, IDA())

    sol_t = sim_trip_gen.solution.t   # Save the time isntants the problem solved at 

    tspan = sim_trip_gen.simulation_inputs.tspan   # Get time span from problem 
    gen_names = [g.name for g in get_components(Generator, sys)]
    deleteat!(gen_names, findall(x->x=="generator-2-Trip",gen_names))
    global_state_index = PSID.get_global_index(sim_trip_gen.simulation_inputs);
    state_index = [get(global_state_index[g], :ω, 0) for g in gen_names]
    N = length(state_index)
    
    Win = randn(N, resSize)'  # Build read in matrix for reservior
    r0 = randn(resSize) # Randomly initialize initial condition of reservoir
    A = erdos_renyi(resSize,resSize, seed=1234)  # Build sparsely connected matrix of reservoir
 
    func(u, p, t) = tanh.(A*u .+ Win*((sim_trip_gen.solution(t)[state_index]) .-1) ./ 0.02)   # Build dynanics of reservoir
    rprob = ODEProblem(func, r0, tspan, nothing) 
    rsol = solve(rprob, Tsit5(), saveat = sim_trip_gen.solution.t)  # Simulate reservoir being driven by nominal soltuion of the system 

    
    # The two parametes are 1) the % of IBR at each node and 2) the % of those IBR that are grid-forming
    param_samples = QuasiMonteCarlo.sample(sample_points, lb, ub, QuasiMonteCarlo.SobolSample()) # Sample parameter sapce

    function get_W(p)
        Gf=p[1]*(1-p[2]) # Grid following %
        GF=p[1]*p[2] # Grid forming %
        
        gen = PSY.get_component(ThermalStandard, sys, "generator-2-Trip")
        PSY.set_available!(gen, true)
        sys=change_ibr_penetration(sys, GF, Gf, ibr_bus, bus_cap, total_power) # Change generation mix at each of the IBR nodes

        sim = PSID.Simulation(
            PSID.ImplicitModel, #Type of model used
            sys,         #system
            file_dir,       #path for the simulation output
            (0.0, 20.0), #time span
            BranchTrip(1.0, "BUS 02-BUS 04-i_4"); # Define perturbation. This is currrently a line-trip but will change
            console_level = Logging.Info,
        ) # Rebuild the system and re-initialize all dynanics states with new IBR %'s

        res = small_signal_analysis(sim)  # Check to ensure system is small-signal stable
        
        gen = PSY.get_component(ThermalStandard, sys, "generator-2-Trip")
        PSY.set_available!(gen, false)
        global_state_index = PSID.get_global_index(sim.simulation_inputs);
        drop_idx=sort(collect(values(global_state_index["generator-2-Trip"])));
        x0_gen_trip = vcat(sim.x0_init[1:drop_idx[1]-1], sim.x0_init[drop_idx[end]+1:end]);
        
        sim_trip_gen = Simulation(
                ImplicitModel,
                sys,
                pwd(),
                (0.0,60.0);
                initialize_simulation = false,
                initial_conditions = x0_gen_trip,
                )
        
        execute!(sim_trip_gen, IDA();)
        
        R = reduce(hcat, rsol.(sol_t))
        S = reduce(hcat, (sim_trip_gen.solution.(sol_t)))[state_index, :]
       
        Wout = (svd(R') \ S')' # calculate read-out matrix
        Wout, R, S
    end

    res = pmap(x -> get_W(param_samples[:,x]), 1:sample_points) # Distribute computaiton among all available workers
    Woutparams = [param_samples[:,i] for i = 1:size(param_samples, 2)] # Distribute computaiton among all available workers
    Wouts = map(x -> x[1], res) # get read-out matrices from returned res object

    f = RadialBasis(Woutparams, Wouts, lb, ub)  # Fit interpolating function used for returning parameter-dependent read-out matrix
    f, rsol, N
end

function predict(p, f, rsol, ts, N, resSize)
    Woutpred = reshape(f(p), N, resSize)
    pred = Woutpred * reduce(hcat, rsol.(ts))
    pred
end

file_dir = joinpath(pwd(), "data",)
include(joinpath(file_dir, "system_models.jl"))
include(joinpath(file_dir, "dynamic_test_data.jl"))

sys, bus_cap = buid_system()   # Build the system

df = solve_powerflow(sys)
total_power=sum(df["bus_results"].P_gen)

generators = [g for g in get_components(Generator, sys)]
for gen in generators
    dyn_gen = dyn_gen_genrou(gen)
    add_component!(sys, dyn_gen, gen);
end

ibr_bus=[3, 6, 8] # Buses to place IBR generation at
GF=0.5*0.15  # % of Grid-forming inverters for nominal case
Gf=0.5*(1-0.15) # % of Grid-following inverters for nominal case

sys = add_ibr(sys, GF, Gf, ibr_bus, bus_cap, total_power)

sample_vals = 40

LB = [0.1, 0.1] # Lower-bounds on the 1) % of IBR at each node and 2) % of those IBR that are grid-forming
UB = [0.7, 1] # Upper-bounds on the 1) % of IBR at each node and 2) % of those IBR that are grid-forming
resSize=1000  # Size of the reservoir

gen = PSY.get_component(ThermalStandard, sys, "generator-2-Trip")
PSY.set_available!(gen, true)
surr, resSol, N, state_index, sol_t = build_surrogate(sys, bus_cap, LB, UB, resSize, sample_vals, total_power) 

Inv = [0.1:0.005:0.8;]
GF = [0.3:0.005:1;]
ts = 0:0.01:60
nadir=zeros(length(GF),length(Inv));
rocof=zeros(length(GF),length(Inv));
settling_time=zeros(length(GF),length(Inv));

for i in 1:length(GF)
    for j in 1:length(Inv) 

        pred = predict([Inv[j], GF[i]], surr, resSol, ts, N, resSize)
        
        reverse_freq = vec(reverse(mean(pred, dims=1)))
        top_margin = reverse_freq[1]+0.000333
        bottom_margin = reverse_freq[1]-0.000333
        rise_time = 60-findfirst((reverse_freq .< bottom_margin) .| (reverse_freq .> top_margin))*0.01
        
        settling_time[i,j]=rise_time
        nadir[i,j]=minimum(pred)
        rocof[i,j]=minimum(diff(pred, dims=2))/0.1
    end
end

CSV.write("results/settling_time.csv", DataFrame(settling_time, :auto), header = false)
CSV.write("results/nadir.csv", DataFrame(nadir, :auto), header = false)
CSV.write("results/rocof.csv", DataFrame(rocof, :auto), header = false)


