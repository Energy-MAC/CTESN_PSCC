include(joinpath(file_dir, "models/dynamic_test_data.jl"))
include(joinpath(file_dir, "models/inverter_models.jl"))

function buid_system(ibr_bus, GF, Gf)

    if isfile(joinpath(file_dir, "models/experimental_system.json"))
        try
            sys = System(joinpath(file_dir, "models/experimental_system.json"), runchecks = false)
        catch e
            @error e
            sys = System(joinpath(file_dir, "models/14bus.raw"))#, joinpath(file_dir, "dyn_data.dyr"))
        end
    else
        sys = System(joinpath(file_dir, "models/14bus.raw"))#, joinpath(file_dir, "dyn_data.dyr"))
    
        set_units_base_system!(sys, "DEVICE_BASE")

        default_generators = [g for g in get_components(Generator, sys)]

        gen = get_component(ThermalStandard, sys, "generator-1-1")
        set_base_power!(gen, 80)
        PSY.set_active_power!(gen, 0.8)

        gen = get_component(ThermalStandard, sys, "generator-2-1")
        set_base_power!(gen, 60)
        PSY.set_active_power!(gen, 0.8)

        gen = get_component(ThermalStandard, sys, "generator-2-Trip")
        set_base_power!(gen, 16)
        PSY.set_active_power!(gen, 0.8)

        gen = get_component(ThermalStandard, sys, "generator-3-1")
        set_base_power!(gen, 60)
        PSY.set_active_power!(gen, 0.8)


        gen = get_component(ThermalStandard, sys, "generator-6-1")
        set_base_power!(gen, 60)
        PSY.set_active_power!(gen, 0.8)

        gen = get_component(ThermalStandard, sys, "generator-8-1")
        set_base_power!(gen, 60)
        PSY.set_active_power!(gen, 0.8)

        for gen in default_generators
            dyn_gen = dyn_gen_genrou(gen)
            add_component!(sys, dyn_gen, gen);
        end

    end
    
    if !isfile(joinpath(file_dir, "models/experimental_system.json"))
        to_json(sys, joinpath(file_dir, "models/experimental_system.json"))
    end

    default_generators = [g for g in get_components(Generator, sys)]
    bus_capacity = Dict()
    total_capacity=0
    for g in default_generators
        bus_capacity[g.bus.name] = get_base_power(g)
        total_capacity = total_capacity + get_base_power(g)
    end

    df = solve_powerflow(sys)
    total_power=sum(df["bus_results"].P_gen)

    sys = add_ibr(sys, GF, Gf, ibr_bus, bus_capacity, total_power)

    set_units_base_system!(sys, "DEVICE_BASE")
    return sys, bus_capacity, total_power
end

function add_ibr(sys, GF, Gf, ibr_bus, bus_capacity, total_power)
    
    trip_pu=0.05
    pf=0.8
    
    set_units_base_system!(sys, "DEVICE_BASE")

    default_generators = [g for g in get_components(Generator, sys)]
    
    for g in default_generators
        if occursin("Trip", g.name)==false
            gen = get_component(ThermalStandard, sys, g.name)
            set_base_power!(gen, bus_capacity[g.bus.name]*(1-GF-Gf-trip_pu))
            set_base_power!(gen.dynamic_injector, bus_capacity[g.bus.name]*(1-GF-Gf-trip_pu))
        end
    end 
    
    
    replace_gens = [g for g in default_generators if g.bus.number in ibr_bus]
    GF_capacity = GF*total_power/pf
    Gf_capacity = Gf*total_power/pf
    
    for g in replace_gens

        storage=add_battery(sys, join(["GF_Battery-", g.bus.number]), g.bus.name, GF_capacity/length(ibr_bus), pf, get_reactive_power(g))
        add_component!(sys, storage)
        inverter=add_grid_forming(storage, GF_capacity/length(ibr_bus))
        add_component!(sys, inverter, storage)

        storage=add_battery(sys, join(["Gf_Battery-", g.bus.number]), g.bus.name, Gf_capacity/length(ibr_bus), pf, get_reactive_power(g))
        add_component!(sys, storage)
        inverter=add_grid_following(storage, Gf_capacity/length(ibr_bus))
        add_component!(sys, inverter, storage)
        
    end

    return sys
end

function change_ibr_penetration(sys, GF, Gf, ibr_bus, bus_capacity, total_power)
    
    set_units_base_system!(sys, "DEVICE_BASE")
    generators = [g for g in get_components(Generator, sys)]
    
    for g in generators
        if occursin("Trip", g.name)==false
            gen = get_component(ThermalStandard, sys, g.name)
            set_base_power!(gen, bus_capacity[g.bus.name]*(1-GF-Gf-0.05))
            set_base_power!(gen.dynamic_injector, bus_capacity[g.bus.name]*(1-GF-Gf-0.05))
        end
    end 
    
    replace_gens = [g for g in generators if g.bus.number in ibr_bus]
    GF_capacity = GF*total_power/0.8
    Gf_capacity = Gf*total_power/0.8
    
    for g in replace_gens
    
        gen = get_component(GenericBattery, sys, join(["GF_Battery-", g.bus.number]))
        set_base_power!(gen, GF_capacity/length(ibr_bus))
        set_base_power!(gen.dynamic_injector, GF_capacity/length(ibr_bus))
            
        gen = get_component(GenericBattery, sys, join(["Gf_Battery-", g.bus.number]))
        set_base_power!(gen, Gf_capacity/length(ibr_bus))
        set_base_power!(gen.dynamic_injector, Gf_capacity/length(ibr_bus))
    end
    
    return sys

end

function dyn_gen_genrou(generator)
    return PSY.DynamicGenerator(
        name = get_name(generator),
        ω_ref = 1.0, #ω_ref
        machine = machine_genrou(), #machine
        shaft = shaft_genrou(), #shaft
        avr = avr_type2(), #avr
        prime_mover = tg_type1(), #tg
        pss = pss_none(),
    ) #pss
end