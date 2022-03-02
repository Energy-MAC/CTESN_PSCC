function simulate_reservoir(sys, resSize, maxStep)
    ### Lines 2-26 are simulating the physical system (Not-ML)
    freq_mean=1
    freq_range=0.02
    
    gen = PSY.get_component(PSY.DynamicGenerator, sys, "generator-2-Trip")
    gen_trip = GeneratorTrip(10.0, gen)
    tspan=(0.0, 70.0)
    sim = Simulation!(
        ResidualModel, #Type of model used
        sys,         #system
        file_dir,       #path for the simulation output
        tspan, #time span
        gen_trip,
    )

    res = small_signal_analysis(sim)
    
    execute!(sim, IDA(), dtmax=maxStep)

    sol_t = unique(sim.results.solution.t)   # Save the time instants the problem solved at (ODE solver chooses these time instants) 
    
    gen_names = [g.name for g in get_components(Generator, sys)]
    deleteat!(gen_names, findall(x->x=="generator-2-Trip",gen_names))
    freqIndex = [sim.results.global_index[g][:ω] for g in gen_names]

    N = length(freqIndex) # Get dimension of state vector we want to predict

    Win = randn(N, resSize)'  # Build read in matrix for reservior
    r0 = randn(resSize) # Randomly initialize initial condition of reservoir
    A = erdos_renyi(resSize,resSize, seed=1234)  # Build sparsely connected matrix of reservoir
 
    func(u, p, t) = tanh.(A*u .+ Win*((sim.results.solution(t)[freqIndex]) .-freq_mean) ./ freq_range)   # Build dynanics of reservoir
    rprob = ODEProblem(func, r0, tspan, nothing) # Build ODE problem
    rsol = solve(rprob, Tsit5())  # Simulate reservoir being driven by nominal soltuion of the system 

    rsol, N, freqIndex, sol_t
end
    
function linear_mapping(sys, busCap, lb, ub, trainSize, totalPower, rSol, stateIndex, simStep)
    
    # The two parametes are 1) the % of IBR at each node and 2) the % of those IBR that are grid-forming 
    boundarySamples=reduce(hcat, vec([[x;y] for x in [lb[1], ub[1]], y in [lb[2], ub[2]]]))
    interiorSamples = QuasiMonteCarlo.sample(trainSize-size(boundarySamples)[2], lb, ub, QuasiMonteCarlo.SobolSample()) # Sample parameter sapce
    trainSamples = [boundarySamples interiorSamples]
    
    gen = PSY.get_component(PSY.DynamicGenerator, sys, "generator-2-Trip")
    genTrip = GeneratorTrip(10.0, gen)
    tspan=(0.0, 70.0)
    
    function get_W(p)
        Gf=p[1]*(1-p[2]) # Grid following %
        GF=p[1]*p[2] # Grid forming %
        
        sys=change_ibr_penetration(sys, GF, Gf, ibrBus, busCap, totalPower) # Change generation mix at each of the IBR nodes
        sim = Simulation!(
            ResidualModel, #Type of model used
            sys,         #system
            file_dir,       #path for the simulation output
            tspan, #time span
            genTrip,
        )

        res = small_signal_analysis(sim)

        execute!(sim, IDA())
        
        R = reduce(hcat, rSol.(simStep))
        S = reduce(hcat, (sim.results.solution.(simStep)))[stateIndex, :]
       
        Wout = (svd(R') \ S')' # calculate read-out matrix
        Wout, R, S
    end

    res = pmap(x -> get_W(trainSamples[:,x]), 1:trainSize) # Distribute computaiton among all available workers
    Woutparams = [trainSamples[:,i] for i = 1:size(trainSamples, 2)] # Distribute computaiton among all available workers
    Wouts = map(x -> x[1], res) # get read-out matrices from returned res object
    Woutparams, Wouts
end

function linear_predict(p, f, rSol, simStep, N, resSize)
    Woutpred = reshape(f(p), N, resSize)
    pred = Woutpred * reduce(hcat, rSol.(simStep))
    pred
end


function nonlinear_mapping(sys, busCap, lb, ub, trainSize, totalPower, rSol, stateIndex, simStep)
    
    # The two parametes are 1) the % of IBR at each node and 2) the % of those IBR that are grid-forming
    boundarySamples=reduce(hcat, vec([[x;y] for x in [lb[1], ub[1]], y in [lb[2], ub[2]]])) # Determinstically sample the corners of the parameter space
    interiorSamples = QuasiMonteCarlo.sample(trainSize-size(boundarySamples)[2], lb, ub, QuasiMonteCarlo.SobolSample()) # Generate quasi-random samples from interior of parameter space
    trainSamples = [boundarySamples interiorSamples] # Concatenate boundary and interior smaple to generate training samples
    
    gen = PSY.get_component(PSY.DynamicGenerator, sys, "generator-2-Trip") # (Not-ML)
    genTrip = GeneratorTrip(10.0, gen) # (Not-ML)
    tspan=(0.0, 70.0) # (Not-ML)
    
    function get_W_rbf(p)
        Gf=p[1]*(1-p[2]) # Grid following % (Not-ML)
        GF=p[1]*p[2] # Grid forming % (Not-ML)
        
        sys=change_ibr_penetration(sys, GF, Gf, ibrBus, busCap, totalPower) # Change generation mix at each of the IBR nodes (Not-ML)
        sim = Simulation!(
            ResidualModel, #Type of model used (Not-ML)
            sys,         #system  (Not-ML)
            file_dir,       #path for the simulation output (Not-ML)
            tspan, #time span (Not-ML)
            genTrip,
        )

        res = small_signal_analysis(sim) # (Not-ML)

        execute!(sim, IDA()) # Simulate the pyhsical system
        
        R = reduce(hcat, rSol.(simStep)) # Get solution of reservoir
        S = reduce(hcat, (sim.results.solution.(simStep)))[stateIndex, :] # Get solution of physical system
       
        UB=vec(maximum(R, dims=2)); # Get maximum of reservoir soltuion across all dimensions -> Needed to build RBF
        LB=vec(minimum(R, dims=2)); # Get maximum of reservoir soltuion across all dimensions -> Needed to build RBF

        Input = [R[:,i] for i in 1:size(R,2)]; # Solution of reservoir as vector of vectors
        Output = [S[:,i] for i in 1:size(S,2)]; # Solution of pysical system as vector of vectors
        
        surr = RadialBasis(Input, Output, LB, UB, rad = linearRadial) # Fit RBF function that maps from solution of reservoir to solution of true system
        beta = surr.coeff # Get the weights of the fitted RBF function
        beta, surr, R, S # Return the weights, surrogate, R and S
    end

    res = pmap(x -> get_W_rbf(trainSamples[:,x]), 1:trainSize) # Distribute computation among all available workers
    Woutparams = [trainSamples[:,i] for i = 1:size(trainSamples, 2)] # Get training samples
    Wouts = map(x -> x[1], res) # get RBF weights from returned res object
    surr=res[1][2] # Get one of the fitted surrogates 
    Woutparams, Wouts, surr
end

function nonlinear_predict(p, betaSurr, Surr, rSol, simStep, N, resSize)
    weights=reshape(betaSurr(p), size(Surr.coeff)[1], size(Surr.coeff)[2]) # Re-shape the weigths to pass to the surrogates
    Surr.coeff = weights # Assign the parameter dependent weights to the RBF
    pred = reduce(hcat, Surr.(rSol(simStep).u)) # Predict the response of the system
    pred
end