import utils
import tqdm
import numpy as np
from datasets import duffing_oscillator,save

def runge_kutta_step(t, y, dt, delta, alpha, beta, gamma, omega):
    """
    Perform a single step of the fourth-order Runge-Kutta method to numerically
    solve the Duffing oscillator differential equation.

    The function calculates the derivatives of displacement (dx/dt) and velocity
    (d²x/dt²) at the given time (t) and current state (y) using the Duffing oscillator
    differential equation:

    d²x/dt² + δ*dx/dt + α*x + β*x³ = γ*cos(ω*t)

    The fourth-order Runge-Kutta method is used to update the displacement and
    velocity values by combining weighted averages of four intermediate steps.

    Parameters:
    - t: Current time at which to evaluate the derivatives.
    - y: List containing the current values of the displacement (x) and velocity (v).
    - dt: Time step size for the integration.
    - delta: Damping coefficient (δ).
    - alpha: Linear stiffness coefficient (α).
    - beta: Nonlinear stiffness coefficient (β).
    - gamma: Amplitude of the external driving force (γ).
    - omega: Angular frequency of the external driving force (ω).

    Returns:
    - List containing the updated values of displacement (x) and velocity (v) after
      a single step of the Runge-Kutta method.
    """
    k1 = duffing_oscillator(t           , y             , delta, alpha, beta, gamma, omega) * dt
    k2 = duffing_oscillator(t + dt / 2  , y + k1 / 2    , delta, alpha, beta, gamma, omega) * dt
    k3 = duffing_oscillator(t + dt / 2  , y + k2 / 2    , delta, alpha, beta, gamma, omega) * dt
    k4 = duffing_oscillator(t + dt      , y + k3        , delta, alpha, beta, gamma, omega) * dt
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def runge_kutta_solve(delta, alpha, beta, gamma, ω, x0, v0, t_span, dt):
    """
    Numerically solve the Duffing oscillator using the fourth-order Runge-Kutta method.

    The function integrates the Duffing oscillator differential equation by applying
    the fourth-order Runge-Kutta method to obtain the displacement and velocity of
    the oscillator over time.

    Parameters:
    - delta: Damping coefficient (δ).
    - alpha: Linear stiffness coefficient (α).
    - beta: Nonlinear stiffness coefficient (β).
    - gamma: Amplitude of the external driving force (γ).
    - ω: Angular frequency of the external driving force (ω).
    - x0: Initial displacement.
    - v0: Initial velocity.
    - t_span: Tuple containing the initial and final time for integration.
    - dt: Time step size for the integration.

    Returns:
    - t_eval: NumPy array containing the time points at which the solution is evaluated.
    - x: NumPy array containing the displacement of the oscillator over time.
    - v: NumPy array containing the velocity of the oscillator over time.
    """
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    n_steps = len(t_eval)

    x = np.zeros(n_steps)
    v = np.zeros(n_steps)

    x[0] = x0
    v[0] = v0

    for i in tqdm.tqdm(range(1, n_steps)):
        t = t_eval[i - 1]
        y = np.array([x[i - 1], v[i - 1]])
        x[i], v[i] = runge_kutta_step(t, y, dt, delta, alpha, beta, gamma, ω)

    return t_eval, x, v

def solve_and_plot(save_npz,
                   δ    :int,
                   α    :int,
                   β    :int,
                   γ    :int,
                   ω    :int,
                   x0   :int,
                   v0   :int,
                   dt   :float,
                   t_span   :tuple ,
                   steps    :int = 1000):
    # Initial state vector
    y0 = [x0, v0]

    # Time span for integration
    t_eval = np.linspace(t_span[0], t_span[1], steps)

    # Numerically solve the Duffing equation
    time, displacement, velocity = runge_kutta_solve(δ, α, β, γ, ω, x0, v0, t_span, dt)
    # Plot the results
    utils.plot_XandV_t(model_name = f"Duffing Oscillator (δ={δ}, α={α}, β={β}, γ={γ}, ω={ω})",
                        time            = time, 
                        displacement    = displacement,
                        velocity        = velocity,
                        delta  = δ,
                        alpha  = α,
                        beta   = β,
                        gamma  = γ,
                        omega  = ω,
                        x_lim  = 100   ,             
                        save_iamge = True)
    # if not save_npz:
    save(time = time,
        x = displacement,
        v = velocity,
        delta = δ,
        alpha = α,
        beta = β,
        gamma = γ,
        omega = ω,
        x0 = x0, v0 = v0,
        t_span = t_span,
        dt = dt,
        )