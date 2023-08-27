import numpy as np

def duffing_equation(t, y, delta, alpha, beta, gamma, omega):
    x, v = y
    dxdt = v
    dvdt = -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
    return np.array([dxdt, dvdt])

def runge_kutta_step(func, t, y, h, *params):
    k1 = h * func(t, y, *params)
    k2 = h * func(t + h/2, y + k1/2, *params)
    k3 = h * func(t + h/2, y + k2/2, *params)
    k4 = h * func(t + h, y + k3, *params)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def solve_duffing_equation(delta, alpha, beta, gamma, omega, t_span, initial_conditions, num_points=1000):
    t_values = np.linspace(t_span[0], t_span[1], num_points)
    h = t_values[1] - t_values[0]
    
    x_values = np.zeros(num_points)
    v_values = np.zeros(num_points)
    
    x_values[0], v_values[0] = initial_conditions
    
    for i in range(1, num_points):
        x_values[i], v_values[i] = runge_kutta_step(duffing_equation, t_values[i-1], np.array([x_values[i-1], v_values[i-1]]), h, delta, alpha, beta, gamma, omega)
    
    return t_values, x_values, v_values, (-delta * v_values - alpha * x_values - beta * x_values**3 + gamma * np.cos(omega * t_values))