import numpy as np
from matplotlib import pyplot as plt
from generator import ULASignalGenerator
from doa import Music, RootMusic

thetas = np.deg2rad([-20, 20, 40, 60])
signal_generator = ULASignalGenerator()
r_sensed = signal_generator.generate(thetas)

##################################################
# music
##################################################
music_engine = Music(steering_provider=signal_generator)
spec, scan = music_engine.get_doa(r=r_sensed)

plt.plot(scan, spec)
plt.show()

##################################################
# root music
##################################################
root_engine = RootMusic(steering_provider=signal_generator)
roots = root_engine.get_doa(r=r_sensed)

phi = np.linspace(0, 2*np.pi, 1000)
unit_circle = np.exp(1j * phi)
plt.figure()
plt.plot(unit_circle.real, unit_circle.imag)
plt.scatter(roots.real, roots.imag)
plt.axhline(0)
plt.axvline(0)
plt.gca().set_aspect('equal', 'box')
plt.xlabel("Real")
plt.ylabel("Imag")
plt.title("Root-MUSIC Roots in Complex Plane")
plt.grid(True)
plt.show()