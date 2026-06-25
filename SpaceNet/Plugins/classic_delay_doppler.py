from SpaceNet.Utils.tf_np_ports import gradient, eigh
from SpaceNet.Plugins.plugin import Link, Plugin, Ports

import numpy as np
import scipy


class ClassicDelayDoppler(Plugin):


    def __init__(self,
                 d_sources: int,
                 observ_ctx,
                 signal_provider,
                 ):
        self.input_ports: Ports = {"r_sensed": Link()}
        self.output_ports: Ports = {
            "tau_est": Link(),
            "omega_est": Link(),
            "tau_grid": Link(),
            "cost_function": Link(),
            }

        self.d_sources       = d_sources
        self.observ_ctx      = observ_ctx
        self.signal_provider = signal_provider


    def execute(self) -> None:
        r_batched = self.input_ports['r_sensed'].value

        tau_batched           = []
        omega_batched         = []
        tau_grid_batched      = []
        cost_function_batched = []

        for r in r_batched:
            tau, omega, grid, cost_function = self._execute_one_step(r)
            tau_batched.append(tau)
            omega_batched.append(omega)
            tau_grid_batched.append(tau)
            cost_function_batched.append(cost_function)

        self.output_ports["tau_est"].value       = np.vstack(tau_batched)
        self.output_ports["omega_est"].value     = np.vstack(omega_batched)
        self.output_ports["tau_grid"].value      = np.vstack(tau_grid_batched)
        self.output_ports["cost_function"].value = np.vstack(cost_function_batched)


    def _execute_one_step(self, r) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        N = r.shape[0]  # samples
        M = r.shape[1]  # antennas

        # --- transform to frequency space ---
        R = np.fft.fftshift(np.fft.fft(r, axis=0), axes=0)  # (NxM)

        # --- covariance over the sample space N ---
        R_cov = R @ R.conj().T / M

        ################################################################
        # MUSIC
        ################################################################
        # --- eigendecomposition ---
        _, Un = scipy.linalg.eigh(a=R_cov, subset_by_index=[0, (N - self.d_sources - 1)])

        # --- MUSIC scan over tau; building the cost function ---
        T_obs = self.observ_ctx.window_length
        tau_grid = np.linspace(0, T_obs, N)
        cost_function = np.zeros_like(tau_grid)
        omega_grid = np.zeros_like(tau_grid)

        # --- signal preparation ---
        fs = self.signal_provider.fs
        f = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))  # (Nx1)
        w = f * 2 * np.pi  # (Nx1)
        s = self.signal_provider.generate(n_samples=N)
        S = np.fft.fftshift(np.fft.fft(s))  # (Nx1)
        dS = gradient(S, w, numpy=True)  # (Nx1)

        # --- compute B because it's not dependent on tau ---
        B_mat = np.real(np.array([[S.conj().T @ S, -S.conj().T @ dS],
                                  [-dS.conj().T @ S, dS.conj().T @ dS]]))

        # --- go over the grid and compute the cost function to be minimized ---
        for i, tau in enumerate(tau_grid):
            # build G(tau)
            v_tau = np.exp(-1j * w * tau)
            G = np.stack([S * v_tau, -dS * v_tau], axis=1)
            A_mat = np.real(G.conj().T @ Un @ Un.conj().T @ G)

            lambdas, gammas = eigh(a=A_mat, b=B_mat, subset_by_index=[0, 1])

            lambda_min       = lambdas[0]
            cost_function[i] = lambda_min
            gamma_min        = gammas[:, 0]
            omega_grid[i]    = gamma_min[1] / gamma_min[0]

        # --- results ---
        idx       = np.argsort(cost_function)[:self.d_sources]
        tau_est   = tau_grid[idx]
        omega_est = omega_grid[idx]

        return tau_est, omega_est, tau_grid, cost_function