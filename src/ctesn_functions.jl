"""
simulate_reservoir(sys, maxStep, perturb, tripName)

Simulates the reservior using a nominal solution of the true physcial system

Inputs
- sys: Physical power system
- maxStep: Maximum simulation step of the adaptive ODE solver
- perturb: System perturbation. This is a loss of generation for this work.
- tripName: Name of the generator to be tripped

Outputs
- rsol: Time series solution of the reservoir
- N: Dimension of state vector to predict
- state_index: Index of states we want to predict
- state_labels: State labels
- sol_t: Vector of timesteps that adaptive ODE solver simulated the system at 
- resSize: Dimension of the reservoir
- res_time: Time take to simulate the reservoir
"""
function simulate_reservoir(sys, maxStep, perturb, tripName)

    sim = Simulation!(
        ResidualModel, 
        sys,         
        file_dir,       
        tspan, 
        perturb,
        all_lines_dynamic = true,
    ) # PowerSimulationsDynamics.jl function for building the power system

    res = small_signal_analysis(sim) # PowerSimulationsDynamics.jl function for checking if system is stable
    
    execute!(sim, IDA(), dtmax=maxStep, enable_progress_bar = false) # PowerSimulationsDynamics.jl function for simualting the system

    sol_t = unique(sim.results.solution.t)  # Save the time instants the problem solved at (ODE solver chooses these time instants) 
    
    # The next block of code gets the indices and labels of all line current, nodal voltage and machine frequency states
    gen_names = [g.name for g in get_components(Generator, sys)]
    deleteat!(gen_names, findall(x->x==tripName,gen_names))
    freqIndex = [sim.results.global_index[g][:ω] for g in gen_names]

    device_labels=keys(sim.results.global_index)
    node_labels= [label for label in device_labels if occursin("V_", label)]
    branch_labels = [label for label in device_labels if occursin("Bus ", label)]
    V_R = [sim.results.global_index[n][:R] for n in node_labels]
    V_I = [sim.results.global_index[n][:I] for n in node_labels]
    I_R = [sim.results.global_index[n][:Il_R] for n in branch_labels]
    I_I = [sim.results.global_index[n][:Il_I] for n in branch_labels]

    state_index=vcat(freqIndex, V_R, V_I, I_R, I_I)
    state_labels=vcat(gen_names.*"_ω", node_labels.*"_R", node_labels.*"_I", branch_labels.*"_R", branch_labels.*"_I")
    solArray=reduce(hcat, sim.results.solution.u)
    stateSol=solArray[state_index, :]

    # Here we calculate the man and standard deviation of the states to normalize as input to the reservoir
    state_mean = mean(stateSol, dims=2)
    state_std = 3 .* std(stateSol, dims=2)

    N = length(state_index) 

    resSize = 2*N

    Win = randn(N, resSize)'  # Build read in matrix for reservior
    r0 = randn(resSize) # Randomly initialize initial condition of reservoir
    A = erdos_renyi(resSize,resSize, seed=1234)  # Build sparsely connected matrix of reservoir
 
    func(u, p, t) = vec(tanh.(A*u .+ Win*((sim.results.solution(t)[state_index] .-state_mean) ./ state_std)))   # Build dynanics of reservoir
    rprob = ODEProblem(func, r0, tspan, nothing) # Build ODE problem
    res_time= @elapsed rsol = solve(rprob, Tsit5())  # Simulate reservoir being driven by nominal soltuion of the system 

    rsol, N, state_index, state_labels, sol_t, resSize, res_time
end

"""
    linear_mapping(sys, busCap, lb, ub, trainSize, totalPower, rSol, stateIndex, simStep)

Learn a linear mapping from the solution of the reservoir to solution of the true system as a function of parameters

Inputs
- sys: Physical power system
- busCap: Generation capacity for each bus in the system
- lb: lower-bound on the parameter space
- ub: upper-bound on the parameter space
- trainSize: Number of true solutions of the system to train on
- totalPower: Total active power of the system
- rSol: Solution of the reservoir
- stateIndex: Index of states we want to predict
- simStep: Vector of timesteps that the nominal solution was solved at

Outputs
- Woutparams: Training parameters
- Wouts: Linear mapping for each parameter set
"""
function linear_mapping(sys, busCap, lb, ub, trainSize, totalPower, rSol, stateIndex, simStep)
    
    # The two parametes are 1) the % of IBR at each node and 2) the % of those IBR that are grid-forming 
    boundarySamples=reduce(hcat, vec([[x;y] for x in [lb[1], ub[1]], y in [lb[2], ub[2]]]))
    interiorSamples = QuasiMonteCarlo.sample(trainSize-size(boundarySamples)[2], lb, ub, QuasiMonteCarlo.LatinHypercubeSample()) # Sample parameter sapce Other = SobolSample()
    trainSamples = [boundarySamples interiorSamples]
    
    gen = PSY.get_component(PSY.DynamicGenerator, sys, "generator-2-Trip")
    genTrip = GeneratorTrip(10.0, gen)
 
    
    function get_W(p)
        
        simResults = simSystem!(sys, p, ibrBus, busCap, totalPower, simStep)
        
        R = reduce(hcat, rSol.(simStep))
        S = simResults[stateIndex, :]
       
        Wout = (svd(R') \ S')' # calculate read-out matrix
        Wout, R, S
    end

    res = pmap(x -> get_W(trainSamples[:,x]), 1:trainSize) # Distribute computaiton among all available workers
    Woutparams = [trainSamples[:,i] for i = 1:size(trainSamples, 2)] # Distribute computaiton among all available workers
    Wouts = map(x -> x[1], res) # get read-out matrices from returned res object
    Woutparams, Wouts
end

"""
    linear_predict(p, f, rSol, simStep, N, resSize)

Predict the response of the physcial system for parameters outside the training set with a linear mapping

Inputs
- p: parameters to predict the reponse of system for
- f: rbf surrogate object
- rsol: solution of the reservoir
- simStep: Vector of timesteps that adaptive ODE solver simulated the system at
- N: Dimension of state vector to predict
- resSize: Dimension of the reservoir

Outputs
- pred: Predicted response of the true physical system
"""
function linear_predict(p, f, rSol, simStep, N, resSize)
    Woutpred = reshape(f(p), N, resSize)
    pred = Woutpred * reduce(hcat, rSol.(simStep))
    pred
end

"""
nonlinear_mapping!(sys, busCap, lb, ub, trainSize, totalPower, rSol, stateIndex, simStep)

Learn a non-linear mapping from the solution of the reservoir to solution of the true system as a funciton of parameters

Inputs
- sys: Physical power system
- busCap: Generation capacity for each bus in the system
- lb: lower-bound on the parameters
- ub: upper-bound on the parameters
- trainSize: Number of true solutions of the system to train on
- totalPower: Total active power of the system
- R: Solution of the reservoir
- stateIndex: Index of states we want to predict
- simStep: Vector of timesteps that the nominal solution was solved at
- perturb: System perturbation. This is a loss of generation in this case
- timeSim=false: Flag used to calcualte training time

Outputs
- Woutparams: Training parameters
- rbf_weights: Weights of RBF that maps from solution of reservoir to solution of system for each parameter set
- surr: One of the fitted rbf funcitons
- psid_times: Optional argument for training time to generate training samples
"""
function nonlinear_mapping!(sys, busCap, ibrBus, lb, ub, trainSize, totalPower, R, stateIndex, simStep, perturb, timeSim=false)
    
    # The two parametes are 1) the % of IBR at each node and 2) the % of those IBR that are grid-forming
    boundarySamples=reduce(hcat, vec([[x;y] for x in [lb[1], ub[1]], y in [lb[2], ub[2]]])) # Determinstically sample the corners of the parameter space
    interiorSamples = QuasiMonteCarlo.sample(trainSize-size(boundarySamples)[2], lb, ub, QuasiMonteCarlo.LatinHypercubeSample()) # Generate quasi-random samples from interior of parameter space
    trainSamples = [boundarySamples interiorSamples] # Concatenate boundary and interior smaple to generate training samples
    
    resUB=vec(maximum(R, dims=2)); # Get maximum of reservoir soltuion across all dimensions -> Needed to build RBF
    resLB=vec(minimum(R, dims=2)); # Get maximum of reservoir soltuion across all dimensions -> Needed to build RBF
    rbfInput = [R[:,i] for i in 1:size(R,2)]; # Solution of reservoir as vector of vectors
    
    function get_W_rbf(p, rbfInput, resLB, resUB, timeSim)
        
        if timeSim==true
            simResults, psid_time = simSystem!(sys, p, ibrBus, busCap, totalPower, simStep, perturb, timeSim)
        else
            simResults = simSystem!(sys, p, ibrBus, busCap, totalPower, simStep, perturb, timeSim)
        end
        
        S = simResults[stateIndex, :] # Get solution of physical system

        Output = [S[:,i] for i in 1:size(S,2)]; # Solution of pysical system as vector of vectors
        
        surr = RadialBasis(rbfInput, Output, resLB, resUB, rad = linearRadial) # Fit RBF function that maps from solution of reservoir to solution of true system
        beta = surr.coeff # Get the weights of the fitted RBF function
        
        if timeSim==true
            return beta, surr, R, S, psid_time # Return the weights, surrogate, R and S
        else
            return beta, surr, R, S # Return the weights, surrogate, R and S
        end
    end

    res = pmap(x -> get_W_rbf(trainSamples[:,x], rbfInput, resLB, resUB, timeSim), 1:trainSize) # Distribute computation among all available workers
    Woutparams = [trainSamples[:,i] for i = 1:size(trainSamples, 2)] # Get training samples
    rbf_weights = map(x -> x[1], res) # get RBF weights from returned res object
    if timeSim==true
        psid_times = map(x -> x[5], res) # get RBF weights from returned res object
    end
    surr=res[1][2] # Get one of the fitted surrogates 
    if timeSim==true
        return Woutparams, rbf_weights, surr, psid_times # Return the weights, surrogate, R and S
    else
        return Woutparams, rbf_weights, surr # Return the weights, surrogate, R and S
    end
end

"""
    nonlinear_predict(p, f, rSol, simStep, N, resSize)

Predict the response of the physcial system for parameters outside the training set with a linear mapping

Inputs
- p: parameters to predict the reponse of system for
- Surr: rbf surrogate object to map from reservoir to soltuion of physcial system
- betaSurr: rbf surrogate object to map from parameters to weights of Surr object
- D: Radial basis function interpolation matrix from Surr
- numSteps: Number of timesteps to predict solution of system for

Outputs
- pred: Predicted response of the true physical system
"""
function nonlinear_predict(p, Surr, betaSurr, D, numSteps)
    weights=reshape(betaSurr(p), size(Surr.coeff)[1], size(Surr.coeff)[2]) # Re-shape the weigths to pass to the surrogates
    pred = D*transpose(weights) # Predict the response of the system
    pred[1:numSteps, :]
end

"""
    simSystem!(sys, params, ibrBus, busCap, totalPower, simStep, perturb, timeSim=false)

Predict the response of the physical system to generate training data

Inputs
- sys: model of power system to simulate
- params: parameters to simualte the response of the system for
- ibrBus: Vector of buses that have converter-interfaced generation
- busCap: Installed generation capacity for each bus
- totalPower: Total active power generation of the system
- simStep: Vector of timesteps the adaptive ODE sovler stepped through
- perturb: Perturbation to the system. A loss of generation in this case.
- timeSim=false: Flag for benchmarking computation time

Outputs
- sol: solution of the system
- timed_exc: vector of timed excutions
"""
function simSystem!(sys, params, ibrBus, busCap, totalPower, simStep, perturb, timeSim=false)
    Gf=params[1]*(1-params[2]) # Grid following %
    GF=params[1]*params[2] # Grid forming %
    
    change_ibr_penetration!(sys, GF, Gf, ibrBus, busCap) # Change generation mix at each of the IBR nodes
    
    build_time = @elapsed sim = Simulation!(
        ResidualModel, #Type of model used
        sys,         #system
        file_dir,       #path for the simulation output
        tspan, #time span
        perturb,
        all_lines_dynamic = true,
    )

    res = small_signal_analysis(sim)

    execute!(sim, IDA(), enable_progress_bar = false)

    total_sim_time = build_time + sim.results.time_log[:timed_solve_time]
    if timeSim==true
        return Array(sim.results.solution(simStep)), [build_time, sim.results.time_log[:timed_solve_time], total_sim_time]
    else
        return Array(sim.results.solution(simStep))
    end
end

"""
    itp(pred, simStep, iterTime)

Interpoalte predicted solution. Used for post-processing to calaculate RoCoF etc.

Inputs
- pred: predicted solution of the system
- simStep: Vector of timesteps corresponding to predicetd solution
- iterTime: Vector of timesteps we want to interpoalte at

Outputs
- interpolated_solution: Interpoalted solution
"""
itp(pred, simStep, iterTime) = LinearInterpolation(simStep, pred).(iterTime)

function resmaplePrediction(predict, simStep, interpolateTime)
    resamplePred=[itp.(eachrow(predict[i]), Ref(simStep), Ref(interpolateTime)) for i in 1:size(predict)[1]]
    output = [transpose(reduce(hcat, resamplePred[i])) for i in 1:size(resamplePred)[1]]
    return output
end 

"""
    getSettlingTime(prediction)

Calculates the frequency settling time

Inputs
- prediction: Predicted solution of the system

Outputs
- settling_time: Estimated frequency settling time
"""
function getSettlingTime(prediction)
    mean_freq = mean(prediction, dims=1)
    reverse_freq = vec(reverse(mean_freq))
    top_margin = reverse_freq[1]+settlingBand
    bottom_margin = reverse_freq[1]-settlingBand
    settling_time=(tStop-tripTime)-findfirst((reverse_freq .< bottom_margin) .| (reverse_freq .> top_margin))*interpolateStep
return settling_time
end

"""
    getSystemProperties(sys)

Return the properties of the power system

Inputs
- sys: Power system model

Outputs
- busCap: Total generation capacity at each bus
- totalGen: Total active power generation
- ibrBus: Vector of buses with inverter-based resources
- ibrGen: Vector of objects of type inverter-based resources
- syncGen: Vector of objects of type synchronous machines 

"""
function getSystemProperties(sys)

    syncGen = collect(get_components(Generator, sys));
    ibrGen =  collect(get_components(GenericBattery, sys));
    allGen = vcat(syncGen, ibrGen);

    ibrBus = unique([gen.bus.number for gen in ibrGen]);
    syncBus = unique([gen.bus.number for gen in syncGen]);
    genBus = unique(vcat(ibrBus, syncBus));

    busCap=Dict(zip(genBus, zeros(length(genBus))))

    for bus in genBus
        busCap[bus] = sum([get_base_power(gen) for gen in allGen if gen.bus.number == bus && occursin("Trip", gen.name)==false ])
    end

    powerfow=solve_powerflow(sys)
    totalGen=sum(powerfow["bus_results"].P_gen)

    return busCap, totalGen, ibrBus, ibrGen, syncGen
end