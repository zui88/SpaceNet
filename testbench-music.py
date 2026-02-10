import numpy as np
from matplotlib import pyplot as plt
from generator import ULASignalGenerator
from doa import Music

thetas = np.deg2rad([-20, 20, 40, 60])
signal_generator = ULASignalGenerator()

music_engine = Music(steering_provider=signal_generator)

r_sensed = signal_generator.generate(thetas)
spec, scan = music_engine.compute_doa(r=r_sensed)

plt.plot(scan, spec)
plt.show()