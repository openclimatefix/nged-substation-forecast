# Forecast building blocks

How NGED assembles different kinds of forecast from the "Lego blocks" OCF delivers.

> **Status: 🚧 Planned.** The normalised forecast is the long-term plan (the early MVP forecasts raw
> MW/MVA — see [delivery tables, Table 1](delivery-tables.md#table-1-power_forecast)). The
> capacity and switching building blocks depend on work scheduled for v0.6 / v0.7. OCF will provide
> example Python code (likely a small package) to demonstrate assembling these forecasts. See the
> [roadmap index](index.md) for status conventions.

---

## The idea

Rather than ship one fixed forecast, OCF delivers a set of **building blocks** so NGED can construct
whichever forecast suits the question they are asking. The two headline forecasts NGED will assemble
are the **normal operation forecast** and the **prevailing conditions forecast**.

The blocks are:

1. **Power forecasts scaled to [−1, +1]** (the [`power_forecast`](delivery-tables.md#table-1-power_forecast)
   table). These **always assume a "normal running arrangement" and perfect health** of generators
   and substations — i.e. a worst-case network-constraint planning view. Producing this
   topology-normalised signal in the presence of historical switching events is the target of the
   [differentiable physics switching state-space model](differentiable-physics.md#9-handling-abnormal-running-arrangements-a-switching-state-space-model)
   (v2 research). "Normal" means:
   - *Substations*: all "normally closed" switches are closed and all "normally open" switches are
     open.
   - *Generators*: the generator is unconstrained by NGED's Automatic Network Management (ANM) and
     operating at full capacity.
2. **Dynamically changing effective capacity of generators** (the
   [`effective_capacity`](delivery-tables.md#table-4-effective_capacity) table). E.g. if a wind
   turbine breaks in a wind farm, we estimate the reduced effective capacity over time.
3. **Switching events** (the [`substation_switching`](delivery-tables.md#table-5-substation_switching)
   table). OCF estimates the amount of power diverted across substations.

---

## Sign convention

- **Substations**: positive = power flowing **towards end-users**; negative = excess generation
  flowing **backwards into the grid**.
- **Customer meters (generators)**: positive = the customer is **generating** power sent to NGED's
  grid; negative = the customer is **consuming** power.

---

## The two forecasts NGED assembles

### Normal Operation Forecast (MW or MVA)

Multiply the [−1, +1] forecast by the asset's **maximum / nominal** capacity:

- *Substations*: × the substation's capacity (for v1, likely just the maximum observed power flow
  through that substation).
- *Generators*: × the **maximum estimated capacity** of that generator.

This answers: *"what would this asset do if it were healthy and the network were in its normal
arrangement?"* — the worst-case view useful for network-constraint planning.

### Prevailing Conditions Forecast (MW or MVA)

This forecast **prevails the most recent conditions**:

- *Generators*: × the **most recently observed effective capacity**.
- *Substations*: prevails the **switching state**.
- *Both*: prevails the `generator_or_circuit_fault` flag.

This answers: *"what will this asset actually do over the next 14 days if current conditions
persist?"*

---

## Worked examples

**A 5 MW solar farm (5 × 1 MW inverters) where 1 inverter failed last month** (effective capacity
reduced to 4 MW):

| Forecast | Behaviour |
|---|---|
| Scaled [−1, +1] | Continues as if the farm is healthy; on a sunny day approaches +1. |
| Normal Operation | Assumes capacity is still **5 MW**. |
| Prevailing Conditions | Assumes the inverter stays broken; uses **4 MW** for the next 14 days. |

**A 12 MW bioenergy generator that ran perfectly 2020–2023, then broke in 2024 and is still
broken**:

| Forecast | Behaviour |
|---|---|
| Scaled [−1, +1] | Predicts +1 for the next 14 days. |
| Normal Operation | Assumes 12 MW capacity → forecasts **12 MW** every timestep. |
| Prevailing Conditions | Assumes it stays broken → predicts **0 MW** every timestep. |

> **Why this matters for "not-on" assets.** One trial-area time series (Boston Biomass Generation,
> ID 19) has not been operational since ~mid-2024 and is now essentially noise. The building-blocks
> approach lets the scaled forecast stay well-behaved while the *prevailing conditions* forecast
> correctly reports ~0 MW.
