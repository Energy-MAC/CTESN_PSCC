using Pkg
Pkg.activate(".")
# pkg.instantiate()
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

    res = small_signal_analysis(sim) # Perform a small signal analysis to make sure model is stable
    gen = PSY.get_component(ThermalStandard, sys, "generator-2-Trip")
    PSY.set_available!(gen, false) # Remove generator that is to be tripped
    
    global_state_index = PSID.get_global_index(sim.simulation_inputs); # Get state-index mapping
    drop_idx=sort(collect(values(global_state_index["generator-2-Trip"]))); # Get index of states of generator that should be dropped
    x0_gen_trip = vcat(sim.x0_init[1:drop_idx[1]-1], sim.x0_init[drop_idx[end]+1:end]); # Drop tripped generator from initial condition vector

    sim_trip_gen = Simulation(
            ImplicitModel,
            sys,
            pwd(),
            (0.0,60.0);
            initialize_simulation = false, # Overwrite PSID initializing conditions
            initial_conditions = x0_gen_trip, # Pass in a vector of initial conditions
            )
    sim_max_step=0:1:60 # Specify the minimum granularity of model output
    execute!(sim_trip_gen, IDA(), tstops=sim_max_step) # Simulate model

    sol_t = sim_trip_gen.solution.t   # Save the time isntants the problem solved at 

    tspan = sim_trip_gen.simulation_inputs.tspan   # Get time span from problem 
    gen_names = [g.name for g in get_components(Generator, sys)] # Get names of synchronous generators on the system
    deleteat!(gen_names, findall(x->x=="generator-2-Trip",gen_names)) # Remove tripped generator from list of synchronous units
    global_state_index = PSID.get_global_index(sim_trip_gen.simulation_inputs); # Get state-index mapping
    state_index = [get(global_state_index[g], :ω, 0) for g in gen_names] # Get index of all synchronous machine frequencies
    
    N = length(state_index) # Numer of CTESN output predicitons
    Win = randn(N, resSize)'  # Build read in matrix for reservior
    r0 = randn(resSize) # Randomly initialize initial condition of reservoir
    A = erdos_renyi(resSize,resSize, seed=1234)  # Build sparsely connected matrix of reservoir
 
    func(u, p, t) = tanh.(A*u .+ Win*((sim_trip_gen.solution(t)[state_index]) .-1) ./ 0.02)   # Build dynanics of reservoir
    rprob = ODEProblem(func, r0, tspan, nothing) # Build problem
    rsol = solve(rprob, Tsit5(), saveat = sim_trip_gen.solution.t)  # Simulate reservoir being driven by nominal soltuion of the system 
    
    # The two parametes are 1) the % of IBR at each node and 2) the % of those IBR that are grid-forming
    param_samples = QuasiMonteCarlo.sample(sample_points, lb, ub, QuasiMonteCarlo.SobolSample()) # Sample parameter sapce using Sobol Sequence

    function get_W(p)
        Gf=p[1]*(1-p[2]) # Grid following %
        GF=p[1]*p[2] # Grid forming %
        
        gen = PSY.get_component(ThermalStandard, sys, "generator-2-Trip")
        PSY.set_available!(gen, true) # Make tripped unit avaiable to initialize system 
        sys=change_ibr_penetration(sys, GF, Gf, ibr_bus, bus_cap, total_power) # Change generation mix at each of the IBR nodes

        sim = PSID.Simulation(
            PSID.ImplicitModel, #Type of model used
            sys,         #system
            file_dir,       #path for the simulation output
            (0.0, 20.0), #time span
            console_level = Logging.Info,
        ) # Rebuild the system and re-initialize all dynanics states with new IBR %'s

        res = small_signal_analysis(sim)  # Check to ensure system is small-signal stable
        
        gen = PSY.get_component(ThermalStandard, sys, "generator-2-Trip")
        PSY.set_available!(gen, false) # Trip unit to cause disturbance
        global_state_index = PSID.get_global_index(sim.simulation_inputs); # Get state-index mapping
        drop_idx=sort(collect(values(global_state_index["generator-2-Trip"]))); # Get index of states of generator that should be dropped
        x0_gen_trip = vcat(sim.x0_init[1:drop_idx[1]-1], sim.x0_init[drop_idx[end]+1:end]); # Drop tripped generator from initial condition vector
        
        sim_trip_gen = Simulation(
                ImplicitModel, #Type of model used
                sys, #system
                pwd(), #path for the simulation output
                (0.0,60.0); #time span
                initialize_simulation = false,
                initial_conditions = x0_gen_trip,
                )
        
        execute!(sim_trip_gen, IDA();) #Simulate the full-order model with the disturbance
        
        R = reduce(hcat, rsol.(sol_t)) # Get solution of reservoir at sampling interval of nominal solution
        S = reduce(hcat, (sim_trip_gen.solution.(sol_t)))[state_index, :] # Get output of interest from solution
       
        Wout = (svd(R') \ S')' # calculate read-out matrix
        Wout, R, S # Return Wout, R, S
    end

    res = pmap(x -> get_W(param_samples[:,x]), 1:sample_points) # Distribute computaiton among all available workers
    Woutparams = [param_samples[:,i] for i = 1:size(param_samples, 2)] # Distribute computaiton among all available workers
    Wouts = map(x -> x[1], res) # get read-out matrices from returned res object

    f = RadialBasis(Woutparams, Wouts, lb, ub)  # Fit interpolating function used for returning parameter-dependent read-out matrix
    f, rsol, N
end

function predict(p, f, rsol, ts, N, resSize) # This the the CTESN prediciton fucntion
    Woutpred = reshape(f(p), N, resSize) # Return estimated read-out matrix from radial basis interpolating function
    pred = Woutpred * reduce(hcat, rsol.(ts))
    pred
end


file_dir = joinpath(abspath(joinpath(pwd(), "..")), "data",)
include(joinpath(file_dir, "dynamic_test_data.jl"))
include(joinpath(file_dir, "inverter_models.jl"))
include(joinpath(file_dir, "system_models.jl"))

resSize=1000 # Size of the reservoir

global sys, bus_cap = buid_system()   # Build the system

df = solve_powerflow(sys)
total_power=sum(df["bus_results"].P_gen) # Caluclate total power coming from generation uits

generators = [g for g in get_components(Generator, sys)] # Get list of generators

for gen in generators
    dyn_gen = dyn_gen_genrou(gen)
    add_component!(sys, dyn_gen, gen); # Attach dynamic injector to all generators
end

ibr_bus=[3, 6, 8] # Buses to place IBR generation at
GF=0.5*0.15  # % of Grid-forming inverters for nominal case
Gf=0.5*(1-0.15) # % of Grid-following inverters for nominal case

global sys = add_ibr(sys, GF, Gf, ibr_bus, bus_cap, total_power) # Add IBR to system

sample_vals = [20:10:70;] # Number of times to sample the parameter space to calculate readout matrices

LB = [0.1, 0.1] # Lower-bounds on the 1) % of IBR at each node and 2) % of those IBR that are grid-forming
UB = [0.7, 0.5] # Upper-bounds on the 1) % of IBR at each node and 2) % of those IBR that are grid-forming

test_size=200 # Test size for benchmarking CTESN performance
test_freq_error=zeros(length(sample_vals),test_size)
test_rocof_error=zeros(length(sample_vals),test_size)
test_nadir_error=zeros(length(sample_vals),test_size)

gen_names = [g.name for g in get_components(Generator, sys)]
deleteat!(gen_names, findall(x->x=="generator-2-Trip",gen_names))

for i in 1:length(sample_vals)
    gen = PSY.get_component(ThermalStandard, sys, "generator-2-Trip")
    PSY.set_available!(gen, true)
    global sys=change_ibr_penetration(sys, GF, Gf, ibr_bus, bus_cap, total_power) # Set IBR% prior to building surrogate 
    surr, resSol, N = build_surrogate(sys, bus_cap, LB, UB, resSize, sample_vals[i], total_power) 

    test_params = QuasiMonteCarlo.sample(test_size, LB, UB, QuasiMonteCarlo.SobolSample()) # Sample parameter sapce

    for j in 1:test_size
        Gf_test=test_params[1,j]*(1-test_params[2,j]) # Grid following %
        GF_test=test_params[1,j]*test_params[2,j] # Grid forming %

        gen = PSY.get_component(ThermalStandard, sys, "generator-2-Trip")
        PSY.set_available!(gen, true)
        global sys=change_ibr_penetration(sys, GF_test, Gf_test, ibr_bus, bus_cap, total_power)

        sim = PSID.Simulation!(
            PSID.ImplicitModel, #Type of model used
            sys,         #system
            file_dir,       #path for the simulation output
            (0.0, 10.0), #time span
            console_level = Logging.Info,
        )

        res = small_signal_analysis(sim) # Make sure the physical system is stable
        gen = PSY.get_component(ThermalStandard, sys, "generator-2-Trip")
        PSY.set_available!(gen, false) # Trip generation unit to cause disturbance
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
        
        execute!(sim_trip_gen, IDA(), saveat=0.01) # Simulate the physical model

        ts = sim_trip_gen.solution.t 

        pred = predict(test_params[:,j], surr, resSol, ts, N, resSize) # Use the CTESN to predict the behavior of the system
        
        global_state_index = PSID.get_global_index(sim_trip_gen.simulation_inputs);
        state_index = [get(global_state_index[g], :ω, 0) for g in gen_names]
        
        sol_array = Array(sim_trip_gen.solution) # Convert physcial solution to an array
        freq_error=sol_array[state_index, :] - pred # Construct matrix of difference between true solution and predicted solution

        
        test_nadir_error[i,j]=minimum(sol_array[state_index, :]) - minimum(pred) # Find difference between lowest predicted frequency and actual lowest frequency
        test_freq_error[i,j]=norm(freq_error, Inf) # Find the largest error
        test_rocof_error[i,j]=(minimum(diff(sol_array[state_index, :], dims=2))-minimum(diff(pred, dims=2)))/0.01 # Find the largest error of numerically estimated rate of change
    end 
end

CSV.write("results/train_size/freq_test_errors.csv", DataFrame(test_freq_error, :auto), header = false)
CSV.write("results/train_size/rocof_test_errors.csv", DataFrame(test_rocof_error, :auto), header = false)
CSV.write("results/train_size/nadir_test_errors.csv", DataFrame(test_nadir_error, :auto), header = false)