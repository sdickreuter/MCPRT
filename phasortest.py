import numpy as np
import cmath
import matplotlib.pyplot as plt


c = 1.0
wl = 1.0
n = 1.0

x = np.linspace(0,5*wl,1000)
f = (c / n) / wl
t = x / (c / n)
phase = 2 * np.pi * f * t
phasor = np.exp(1j * phase)

#plt.plot(np.angle(phasor))
plt.plot(phasor.real)
plt.show()