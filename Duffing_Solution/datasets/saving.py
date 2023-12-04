import os
import numpy as np

def save(time,x,v,
         delta,alpha, beta, gamma, omega,
         x0, v0,
         t_span,
         dt)->None:
   
    data_filename = os.path.join('datasets','offline',
                                 f"Duffing Oscillator (δ={delta}, α={alpha}, β={beta}, γ={gamma}, ω={omega} ,{t_span=}).npz")
    
    np.savez_compressed(data_filename,
                        time=time,displacement = x,velocity = v,
                        delta = delta, alpha = alpha, beta = beta, gamma = gamma, omega = omega,
                        x0 = x0, v0 = v0,
                         dt = dt
                        )

