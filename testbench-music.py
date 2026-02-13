import numpy as np
from generator import ULASignalGenerator
from doa import Music, RootMusic, ClassicMusic

thetas = np.deg2rad([-20, 20, 40, 60])
signal_generator = ULASignalGenerator(n_antennas=10)
r_sensed = signal_generator.generate(thetas, n_samples=50, snr_db=10)

##################################################
# music
##################################################
music_engine = ClassicMusic(steering_provider=signal_generator)
est_thetas = music_engine.get_doa(r=r_sensed)
print(est_thetas)

##################################################
# root music
##################################################
root_engine = RootMusic(steering_provider=signal_generator)
est_thetas = root_engine.get_doa(r=r_sensed)
print(est_thetas)