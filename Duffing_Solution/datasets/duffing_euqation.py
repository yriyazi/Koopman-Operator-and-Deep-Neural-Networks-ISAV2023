import numpy as np

def duffing_oscillator(t, y, delta, alpha, β, gamma, ω):
    """
    Function that represents the Duffing oscillator differential equation.

    The Duffing oscillator is a second-order nonlinear differential equation
    that describes the behavior of a damped-driven oscillator. It is governed
    by the equation:

    d²x/dt² + δ*dx/dt + α*x + β*x³ = γ*cos(ω*t)

    where:
    - x: Displacement of the oscillator from its equilibrium position.
    - t: Time.
    - v: Velocity of the oscillator (dx/dt).
    - δ: Damping coefficient, determines the rate of energy dissipation.
    - α: Linear stiffness coefficient.
    - β: Nonlinear stiffness coefficient.
    - γ: Amplitude of the external driving force.
    - ω: Angular frequency of the external driving force.

    Parameters:
    - t: Time at which to evaluate the derivatives.
    - y: List containing the current values of the displacement (x) and velocity (v).
    - delta: Damping coefficient (δ).
    - alpha: Linear stiffness coefficient (α).
    - β: Nonlinear stiffness coefficient.
    - gamma: Amplitude of the external driving force (γ).
    - ω: Angular frequency of the external driving force.

    Returns:
    - List containing the derivatives of displacement (dx/dt) and velocity (d²x/dt²)
      at the given time (t).
    """
    x, v = y
    dxdt = v
    dvdt = -delta * v - alpha * x - β * x**3 + gamma * np.cos(ω * t)
    return np.array([dxdt, dvdt])
