function add_grid_forming(storage, capacity)
    return DynamicInverter(
        name = get_name(storage),
        ω_ref = 1.0, # ω_ref,
        converter = AverageConverter(rated_voltage = 138.0, rated_current = (capacity*1e3)/138.0), #converter
        outer_control = outer_control_droop(), #outer control
        inner_control = inner_control(), #inner control voltage source
        dc_source = FixedDCSource(voltage = 600.0), #dc source
        freq_estimator = no_pll(), #pll
        filter = filt(), #filter
    )
end

function add_grid_following(storage, capacity)
    return DynamicInverter(
        name = get_name(storage),
        ω_ref = 1.0, # ω_ref,
        converter = AverageConverter(rated_voltage = 138.0, rated_current = (capacity*1e3)/138.0), #converter
        outer_control = outer_control_gfoll(), #ogetuter control
        inner_control = current_mode_inner(), #inner control voltage source
        dc_source = FixedDCSource(voltage = 600.0), #dc source
        freq_estimator = pll(), #pll
        filter = filt(), #filter
    )
end

function add_battery(sys, battery_name, bus_name, capacity, P, Q)
    return GenericBattery(
        name = battery_name,
        bus = get_component(Bus, sys, bus_name),
        available = true,
        prime_mover = PrimeMovers.BA,
        active_power = P,
        reactive_power = Q,
        rating = 1.1,
        base_power = capacity,
        initial_energy = 50.0,
        state_of_charge_limits = (min = 5.0, max = 100.0),
        input_active_power_limits = (min = 0.0, max = 1.0),
        output_active_power_limits = (min = 0.0, max = 1.0),
        reactive_power_limits = (min = -1.0, max = 1.0),
        efficiency = (in = 0.80, out = 0.90),
    )
end

function dyn_gen_genrou(generator, H, D)
    return PSY.DynamicGenerator(
        name = get_name(generator),
        ω_ref = 1.0, #ω_ref
        machine = machine_genrou(), #machine
        shaft = heterogeneous_shaft(H, D), #shaft
        avr = avr_type2(), #avr
        prime_mover = tg_type1(), #tg
        pss = pss_none(),
    ) #pss
end

function add_Thermal(sys, name, bus_name, capacity, P, Q)
    return ThermalStandard(
        name= name,
        available= true,
        status= true,
        bus = get_component(Bus, sys, bus_name),
        active_power= P,
        reactive_power= Q,
        rating= 39.61163465447999,
        active_power_limits= (min = 0.0, max = 0.96),
        reactive_power_limits= (min = -39.6, max = 39.6),
        ramp_limits= (up = 2.4, down = 2.4),
        base_power= capacity,
        operation_cost=ThreePartCost(nothing),
    )
end

heterogeneous_shaft(H, D) = SingleMass(
    H,
    D,
)

function dyn_gen_second_order(generator, H, D)
    return DynamicGenerator(
        name = get_name(generator),
        ω_ref = 1.0, # ω_ref,
        machine = machine_anderson(), #machine
        shaft = heterogeneous_shaft(H, D), #shaft
        avr = avr_type1(), #avr
        prime_mover = tg_type1(), #tg
        pss = pss_none(), #pss
    )
end