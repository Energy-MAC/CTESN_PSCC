file_dir = joinpath(
    pwd(),
    "data",
)

include(joinpath(file_dir, "inverter_models.jl"))
include(joinpath(file_dir, "dynamic_test_data.jl"))

function buid_system(Gf, GF, ibr_bus)

    sys = System(joinpath(file_dir, "14bus.raw"), joinpath(file_dir, "dyn_data.dyr"))
    set_units_base_system!(sys, "DEVICE_BASE")

    default_generators = [g for g in get_components(Generator, sys)]

    gen = get_component(ThermalStandard, sys, "generator-1-1")
    set_base_power!(gen, 180)

    gen = get_component(ThermalStandard, sys, "generator-2-1")
    set_base_power!(gen, 100)

    gen = get_component(ThermalStandard, sys, "generator-3-1")
    set_base_power!(gen, 140)


    gen = get_component(ThermalStandard, sys, "generator-6-1")
    set_base_power!(gen, 100)

    gen = get_component(ThermalStandard, sys, "generator-8-1")
    set_base_power!(gen, 140)

    bus_capacity = Dict()
    for g in default_generators
        bus_capacity[g.bus.name] = get_base_power(g)
    end

    replace_gens = [g for g in default_generators if g.bus.number in ibr_bus]
    
    for g in replace_gens
        gen = get_component(ThermalStandard, sys, g.name)
        set_base_power!(gen, bus_capacity[g.bus.name]*(1-GF-Gf))

        storage=add_battery(sys, join(["GF_Battery-", g.bus.number]), g.bus.name, bus_capacity[g.bus.name]*(GF), get_active_power(g), get_reactive_power(g))
        add_component!(sys, storage)
        inverter=add_grid_forming(storage, bus_capacity[g.bus.name]*(GF))
        add_component!(sys, inverter, storage)
        @show get_active_power(storage)

        storage=add_battery(sys, join(["Gf_Battery-", g.bus.number]), g.bus.name, bus_capacity[g.bus.name]*(Gf), get_active_power(g), get_reactive_power(g))
        add_component!(sys, storage)
        inverter=add_grid_following(storage, bus_capacity[g.bus.name]*(Gf))
        add_component!(sys, inverter, storage)
    end

    return sys, bus_capacity
end

function change_ibr_penetration(Gf, GF, ibr_bus, bus_capacity, sys)
    
    generators = [g for g in get_components(Generator, sys)]
    replace_gens = [g for g in generators if g.bus.number in ibr_bus]

    for g in replace_gens
        gen = get_component(ThermalStandard, sys, g.name)
        set_base_power!(gen, bus_capacity[g.bus.name]*(1-GF-Gf))
    
        gen = get_component(GenericBattery, sys, join(["GF_Battery-", g.bus.number]))
        gen.base_power=bus_capacity[g.bus.name]*(GF)
        
        gen = get_component(GenericBattery, sys, join(["Gf_Battery-", g.bus.number]))
        gen.base_power=bus_capacity[g.bus.name]*(Gf)
    end

end