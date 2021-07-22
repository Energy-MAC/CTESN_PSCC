function add_grid_forming(storage, capacity)
    return DynamicInverter(
        name = get_name(storage),
        ω_ref = 1.0, # ω_ref,
        converter = AverageConverter(rated_voltage = 138.0, rated_current = (capacity*1e3)/138.0), #converter
        outer_control = outer_control(), #outer control
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
        outer_control = outer_control_droop(), #ogetuter control
        inner_control = inner_control(), #inner control voltage source
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