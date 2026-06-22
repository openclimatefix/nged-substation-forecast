## Project Summary: Differentiable Physics (DP) for NGED

This framework bridges pure machine learning with mechanistic domain knowledge to solve the Net Demand Disaggregation problem across the National Grid Electricity Distribution (NGED) network.

---

## 1. Why Differentiable Physics?

Pure black-box models (like standard Neural Networks or XGBoost) require vast amounts of data to learn basic physical invariants and are prone to overfitting or hallucinating non-physical behavior. We inject a **Differentiable Physics (DP)** layer directly into our computational graph for three core reasons:

* **Data and Sample Efficiency:** The model does not have to spend capacity learning that the sun rises in the east or that wind power scales cubically with velocity ($P \propto v^3$). The physics engine provides these constraints for free, allowing the ML components to focus entirely on atmospheric and behavioral nuances.
* **True Invertibility (Inputs as Parameters):** Because every operation in our physics engine preserves gradients, we can backpropagate errors all the way to the *inputs*. This lets us treat unobserved variables (like local irradiance) or system configurations as parameters that can be solved via gradient descent.
* **Guaranteed Interpretability:** Instead of inspecting uninterpretable latent layers, our system updates explicit physical parameters like tilt, azimuth, or capacity. This allows engineers to immediately audit the model's assumptions.

---

## 2. Python Sketch: `DifferentiableSolarPlant`

To account for configuration uncertainties, we model each physical parameter as a learnable Normal distribution $\mathcal{N}(\mu, \sigma^2)$. We utilize the reparameterization trick (`rsample()`) to ensure that gradients pass cleanly through the stochastic sampling process during end-to-end training.

```python
import torch
import torch.nn as nn
import torch.distributions as dist

class DifferentiableSolarPlant(nn.Module):
    def __init__(self, init_tilt_mu: float, init_azimuth_mu: float, init_capacity_mu: float):
        super().__init__()
        # Represent parameters as Normal distributions using variational parameters (mu and log_std)
        self.tilt_mu = nn.Parameter(torch.tensor(init_tilt_mu))
        self.tilt_log_std = nn.Parameter(torch.tensor(-2.0))  # exp(-2) ~ 0.13 std initially
        
        self.azimuth_mu = nn.Parameter(torch.tensor(init_azimuth_mu))
        self.azimuth_log_std = nn.Parameter(torch.tensor(-2.0))
        
        self.capacity_mu = nn.Parameter(torch.tensor(init_capacity_mu))
        self.capacity_log_std = nn.Parameter(torch.tensor(-2.0))

    def get_parameter_distributions(self):
        """Returns the policy distributions for the physical parameters."""
        tilt_dist = dist.Normal(self.tilt_mu, torch.exp(self.tilt_log_std))
        azimuth_dist = dist.Normal(self.azimuth_mu, torch.exp(self.azimuth_log_std))
        capacity_dist = dist.Normal(self.capacity_mu, torch.exp(self.capacity_log_std))
        return tilt_dist, azimuth_dist, capacity_dist

    def forward(self, ghi: torch.Tensor, sun_zenith_rad: torch.Tensor, sun_azimuth_rad: torch.Tensor) -> torch.Tensor:
        """
        Computes power output by sampling from the parameter distributions.
        All inputs and operations preserve gradients.
        """
        tilt_dist, azimuth_dist, capacity_dist = self.get_parameter_distributions()
        
        # Sample using the reparameterization trick to allow backpropagation
        tilt = tilt_dist.rsample()
        azimuth = azimuth_dist.rsample()
        capacity = capacity_dist.rsample()
        
        # Enforce physical boundaries
        tilt = torch.clamp(tilt, min=0.0, max=torch.pi/2)
        capacity = torch.clamp(capacity, min=0.0)
        
        # Physics: Calculate Angle of Incidence (AOI) on the tilted panel
        # Cosine law of tracking/fixed surfaces
        cos_aoi = (torch.cos(sun_zenith_rad) * torch.cos(tilt) + 
                   torch.sin(sun_zenith_rad) * torch.sin(tilt) * torch.cos(sun_azimuth_rad - azimuth))
        
        cos_aoi = torch.clamp(cos_aoi, min=0.0)  # No power if sun is behind panels
        
        # Basic physical conversion (Plane of Array Irradiance * Effective Capacity)
        poa_irradiance = ghi * cos_aoi
        predicted_power = poa_irradiance * capacity
        
        return predicted_power

```

---

## 3. Graph Neural Network (GNN) Integration

The distribution network is fundamentally a topological graph. We couple our Differentiable Physics engine with a spatial GNN to ensure that power balances across the grid are strictly conserved.

### Node Definitions

The GNN models the distribution network topology using six distinct node types:

1. **Substations:** Measure the empirical, net blended power flow (the main target constraint).
2. **Metered PV Sites:** Commercial arrays outputting live, dedicated generation data.
3. **Metered Wind Sites:** Co-located or standalone transmission-metered wind assets.
4. **Unmetered PV Fleets:** Aggregated behind-the-meter (BTM) solar arrays grouped by location.
5. **Unmetered Wind Fleets:** Micro-generation or distributed wind assets lacking SCADA visibility.
6. **Gross Demand Nodes:** The true, underlying consumer load (inferred).

### The Fusion Mechanism

The GNN handles spatial message-passing to capture regional correlations (e.g., if it is raining at Substation A, it is likely cloudy at adjacent Unmetered PV Fleet B).

The output of the GNN's spatial hidden layers feeds directly into the parameters/inputs of our **DP Modules**. The DP modules then calculate the explicit physical power generation. Finally, a hard physical Kirchhoff constraint node aggregates the elements:

$$\text{Net Substation Demand} = \text{Gross Demand} - (\text{PV}_{\text{metered}} + \text{PV}_{\text{unmetered}}) - (\text{Wind}_{\text{metered}} + \text{Wind}_{\text{unmetered}})$$

The error between predicted net demand and actual substation net demand produces a gradient that flows backward through the graph structure, optimizing both the GNN weights and the physical parameter distributions simultaneously.

---

## 4. Phasing Roadmap

To minimize engineering risk, the project will be built and validated in two sequential phases:

### Phase 1: Dynamic Capacity Estimation (v1)

Before attempting full disaggregation, DP will be deployed purely to estimate capacity of the **Metered PV and Wind sites**.

* **The Problem:** Generator capacity changes continuously due to maintenance, inverter dropouts, turbine faults, or curtailment. Static capacity values introduce massive downstream errors.
* **The Solution:** By locking the known physical coordinates and passing live weather data through the DP module, any residual delta between expected power and actual power is backpropagated to dynamically update the **Effective Capacity parameter** ($\mu_{\text{capacity}}$) on a rolling basis. This acts as a real-time health and availability monitor for the grid.

### Phase 2: Full GNN-Coupled Disaggregation (v2)

Once the metered assets are accurately tracked, we unlock the full architecture. The DP modules for the *unmetered* fleets are bound to the GNN nodes. The network uses the verified metered assets and spatial weather cues to disaggregate the mixed substation signals, cleanly separating true gross demand from hidden BTM renewable generation.

---

## 5. Long-Term Vision: GB-Wide Inverse Irradiance Mapping

Because the v1 architecture establishes highly calibrated, parameter-verified "virtual sensors" across the metered fleet, we can execute the inversion trick at scale. By freezing the calibrated asset parameters and running gradient descent backward through the DP components based on live generation data, we can dynamically extract a high-resolution, dense grid of surface irradiance and wind-field estimates across the entirety of Great Britain. This will serve as an independent, high-frequency, physics-validated weather product for real-time grid balancing.
