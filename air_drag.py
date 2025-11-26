import numpy as np
import matplotlib.pyplot as plt

# ----------------- TENNIS BALL (official values) -----------------
m   = 0.057                     # kg (official mass)
g   = 9.81
rho = 1.2
r   = 0.0335                     # radius = 3.35 cm
A   = np.pi * r**2               # exact area
Cd  = 0.5

y = 20.0                         # drop height
v = 0.0
t = 0.0
dt = 0.005                       # smaller step = perfect accuracy

time_list = [t]
height    = [y]
velocity  = [v]

while y > 0:
    # Save current state
    time_list.append(t)
    height.append(y)
    velocity.append(v)
    
    # FULL RK4 for BOTH position and velocity
    k1y = v
    k1v = -g - (0.5*rho*A*Cd*v*abs(v))/m

    k2y = v + 0.5*dt*k1v
    k2v = -g - (0.5*rho*A*Cd*k2y*abs(k2y))/m

    k3y = v + 0.5*dt*k2v
    k3v = -g - (0.5*rho*A*Cd*k3y*abs(k3y))/m

    k4y = v + dt*k3v
    k4v = -g - (0.5*rho*A*Cd*k4y*abs(k4y))/m

    v = v + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v)
    y = y + (dt/6)*(k1y + 2*k2y + 2*k3y + k4y)
    t = t + dt

    if y <= 0:
        height[-1] = 0
        break
v_terminal = np.sqrt(2*m*g / (rho*A*Cd))

print(f"Theoretical terminal velocity = {v_terminal:.1f} m/s")
print(f"Impact speed (simulation)      = {abs(v):.1f} m/s")
print(f"Time to ground                 = {t:.2f} s")
print(f"No-drag impact speed would be  = {np.sqrt(2*g*20):.1f} m/s")

# ----------------- PERFECT PLOT -----------------
plt.figure(figsize=(11,4.5))

plt.subplot(1,2,1)
plt.plot(time_list, velocity, 'orange', lw=3)
plt.axhline(-v_terminal, color='red', linestyle='--', linewidth=2, label=f'Terminal = {v_terminal:.1f} m/s')
plt.title("Tennis Ball Drop – Velocity", fontsize=14)
plt.xlabel("Time (s)"); plt.ylabel("Velocity (m/s)")
plt.grid(alpha=0.3); plt.legend()

plt.subplot(1,2,2)
plt.plot(time_list, height, 'green', lw=3)
plt.title("Tennis Ball Drop – Height", fontsize=14)
plt.xlabel("Time (s)"); plt.ylabel("Height (m)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()