function simulate_reservoir(sys, resSize, maxStep)
    
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

    sol_t = unique(sim.results.solution.t)   # Save the time isntants the problem solved at 
    
    gen_names = [g.name for g in get_components(Generator, sys)]
    deleteat!(gen_names, findall(x->x=="generator-2-Trip",gen_names))
    freqIndex = [sim.results.global_index[g][:ω] for g in gen_names]

    N = length(freqIndex)

    Win = randn(N, resSize)'  # Build read in matrix for reservior
    r0 = randn(resSize) # Randomly initialize initial condition of reservoir
    A = erdos_renyi(resSize,resSize, seed=1234)  # Build sparsely connected matrix of reservoir
 
    func(u, p, t) = tanh.(A*u .+ Win*((sim.results.solution(t)[freqIndex]) .-freq_mean) ./ freq_range)   # Build dynanics of reservoir
    rprob = ODEProblem(func, r0, tspan, nothing) 
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