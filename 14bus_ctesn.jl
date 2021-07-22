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
gr()

function build_surrogate(sys, bus_cap, lb, ub, resSize, sample_points)

    sim = PSID.Simulation(
        PSID.ImplicitModel, #Type of model used
        sys,         #system
        file_dir,       #path for the simulation output
        (0.0, 20.0), #time span
        BranchTrip(1.0, "BUS 02-BUS 04-i_4");
        console_level = Logging.Info,
    )
    
    res = small_signal_analysis(sim)
    
    PSID.execute!(sim, IDA(); abstol = 1e-8)

    sol_t = sim.solution.t   # Save the time isntants the problem solved at 

    tspan = sim.simulation_inputs.tspan   # Get time span from problem 
    N = size(sim.solution.u[1], 1) # Get number of states

    Win = randn(N, resSize)'  # Build read in matrix for reservior
    r0 = randn(resSize) # Randomly initialize initial condition of reservoir
    A = erdos_renyi(resSize,resSize)  # Build sparsely connected matrix of reservoir
 
    func(u, p, t) = tanh.(A*u .+ Win*(sim.solution(t)))  # Build dynanics of reservoir
    rprob = ODEProblem(func, r0, tspan, nothing) 
    rsol = solve(rprob, Tsit5(), saveat = sim.solution.t)  # Simulate reservoir being driven by nominal soltuion of the system 

    # The two parametes are 1) the % of IBR at each node and 2) the % of those IBR that are grid-forming
    param_samples = QuasiMonteCarlo.sample(sample_points, lb, ub, QuasiMonteCarlo.SobolSample()) # Randomly sample parameter sapce

    function get_W(p)
        Gf=p[1]*(1-p[2]) # Grid following %
        GF=p[1]*p[2] # Grid forming %

        change_ibr_penetration(Gf, GF, ibr_bus, bus_cap, sys) # Change generation mix at each of the IBR nodes

        sim = PSID.Simulation(
            PSID.ImplicitModel, #Type of model used
            sys,         #system
            file_dir,       #path for the simulation output
            (0.0, 20.0), #time span
            BranchTrip(1.0, "BUS 02-BUS 04-i_4"); # Define perturbation. This is currrently a line-trip but will change
            console_level = Logging.Info,
        ) # Rebuild the system and re-initialize all dynanics states with new IBR %'s

        res = small_signal_analysis(sim)  # Check to ensure system is small-signal stable

        PSID.execute!(sim, IDA(); abstol = 1e-8) # Simulate system

        R = reduce(hcat, rsol.(sol_t))
        S = reduce(hcat, (sim.solution.(sol_t)))
       
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

ibr_bus=[3, 6, 8] # Buses to place IBR generation at
GF=0.2 # % of Grid-forming inverters for nominal case
Gf=0.2 # % of Grid-following inverters for nominal case

sys, bus_cap = buid_system(Gf, GF, ibr_bus)   # Build the system

numSamples = 2 # Number of times to sample the parameter space to calcualte readout matrices

LB = [0.1, 0.1] # Lower-bounds on the 1) % of IBR at each node and 2) % of those IBR that are grid-forming
UB = [0.8, 1] # Upper-bounds on the 1) % of IBR at each node and 2) % of those IBR that are grid-forming
resSize=3000  # Size of the reservoir

surr, resSol, N = build_surrogate(sys, bus_cap, LB, UB, resSize, numSamples)

ts = 0:0.1:20  
test_params = [0.4, 0.5] # New test parameters 

pred = predict(test_params, surr, resSol, ts, N, resSize)
