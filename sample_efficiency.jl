using Pkg
Pkg.activate(".")
using PowerSimulationsDynamics
PSID = PowerSimulationsDynamics
using PowerSystems
using Logging
using Sundials
using Plots
using LightGraphs
using OrdinaryDiffEq
using QuasiMonteCarlo
using LinearAlgebra
using Distributed
using Surrogates
using CSV
using DataFrames
gr()

function build_surrogate(sys, bus_cap, lb, ub, resSize, sample_points, total_power)
    
    sim = PSID.Simulation!(
        PSID.ImplicitModel, #Type of model used
        sys,         #system
        file_dir,       #path for the simulation output
        (0.0, 10.0), #time span
        BranchTrip(1.0, "BUS 02-BUS 04-i_4");
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
            (0.0,30.0);
            initialize_simulation = false,
            initial_conditions = x0_gen_trip,
            )

    execute!(sim_trip_gen, IDA();)

    sol_t = sim_trip_gen.solution.t   # Save the time isntants the problem solved at 

    tspan = sim_trip_gen.simulation_inputs.tspan   # Get time span from problem 
    N = size(sim_trip_gen.solution.u[1], 1) # Get number of states

    Win = randn(N, resSize)'  # Build read in matrix for reservior
    r0 = randn(resSize) # Randomly initialize initial condition of reservoir
    A = erdos_renyi(resSize,resSize)  # Build sparsely connected matrix of reservoir
 
    func(u, p, t) = tanh.(A*u .+ Win*(sim_trip_gen.solution(t)))  # Build dynanics of reservoir
    rprob = ODEProblem(func, r0, tspan, nothing) 
    rsol = solve(rprob, Tsit5(), saveat = sim_trip_gen.solution.t)  # Simulate reservoir being driven by nominal soltuion of the system 

    # The two parametes are 1) the % of IBR at each node and 2) the % of those IBR that are grid-forming
    param_samples = QuasiMonteCarlo.sample(sample_points, lb, ub, QuasiMonteCarlo.SobolSample()) # Randomly sample parameter sapce

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
        S = reduce(hcat, (sim_trip_gen.solution.(sol_t)))
       
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

function dyn_gen_genrou(generator)
    return PSY.DynamicGenerator(
        name = get_name(generator),
        ω_ref = 1.0, #ω_ref
        machine = machine_genrou(), #machine
        shaft = shaft_genrou(), #shaft
        avr = avr_type1(), #avr
        prime_mover = tg_type1(), #tg
        pss = pss_none(),
    ) #pss
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
GF=0.1 # % of Grid-forming inverters for nominal case
Gf=0.6 # % of Grid-following inverters for nominal case

sys = add_ibr(sys, GF, Gf, ibr_bus, bus_cap, total_power)

sample_vals = [10:10:100;] # Number of times to sample the parameter space to calculate readout matrices

LB = [0.1, 0.1] # Lower-bounds on the 1) % of IBR at each node and 2) % of those IBR that are grid-forming
UB = [0.7, 1] # Upper-bounds on the 1) % of IBR at each node and 2) % of those IBR that are grid-forming
resSize=3000  # Size of the reservoir

test_size =20
test_error=zeros(length(sample_vals),test_size)

for i in 1:length(sample_vals)
    surr, resSol, N = build_surrogate(sys, bus_cap, LB, UB, resSize, sample_vals[i], total_power) 

    test_params = [0.7 .* rand(test_size) .+ 0.1 0.9 .* rand(test_size) .+ 0.1]

    for j in 1:test_size
        Gf_test=test_params[j,1]*(1-test_params[j,2]) # Grid following %
        GF_test=test_params[j,1]*test_params[j,2] # Grid forming %

        gen = PSY.get_component(ThermalStandard, sys, "generator-2-Trip")
        PSY.set_available!(gen, true)
        sys=change_ibr_penetration(sys, GF_test, Gf_test, ibr_bus, bus_cap, total_power)

        sim = PSID.Simulation!(
            PSID.ImplicitModel, #Type of model used
            sys,         #system
            file_dir,       #path for the simulation output
            (0.0, 10.0), #time span
            BranchTrip(1.0, "BUS 02-BUS 04-i_4");
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
        
        execute!(sim_trip_gen, IDA();)

        ts = sim_trip_gen.solution.t 

        pred = predict(test_params[j,:], surr, resSol, ts, N, resSize)
        println("Here")
        gen_names = [g.name for g in get_components(Generator, sys)]
        deleteat!(gen_names, findall(x->x=="generator-2-Trip",gen_names))
        
        global_state_index = PSID.get_global_index(sim_trip_gen.simulation_inputs);
        state_index = [get(global_state_index[g], :ω, 0) for g in gen_names]
        
        sol_array = Array(sim_trip_gen.solution)
        freq_error=sol_array[state_index, :] - pred[state_index, :]
        
        test_error[i,j]=norm(freq_error, Inf)
    end 
end

CSV.write("results/test_errors.csv", DataFrame(test_error, :auto), header = false)