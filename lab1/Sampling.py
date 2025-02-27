import numpy as np
import matplotlib.pyplot as plt

# Given parameters
F1 = 300e6  # Frequency in Hz
Fs = 1000e6  # Sampling frequency in Hz
Ts = 1 / Fs  # Sampling period
T = 10 / F1  # Duration covering 10 cycles of the cosine wave
t = np.linspace(0, T, 1000)  # High-resolution time vector

# Original signal
x_t = np.cos(2 * np.pi * F1 * t)

# Sample points
n = np.arange(0, T, Ts)
x_n = np.cos(2 * np.pi * F1 * n)

# Shifted sample points (Ts/2 shift)
n_shifted = n + Ts / 2
x_n_shifted = np.cos(2 * np.pi * F1 * n_shifted)

# Reconstruction using sinc interpolation
def sinc_interp(x_n, n, t, Ts):
    return np.sum(x_n[:, None] * np.sinc((t - n[:, None]) / Ts), axis=0)

# Reconstructed signals
x_r_t = sinc_interp(x_n, n, t, Ts)
x_r_t_shifted = sinc_interp(x_n_shifted, n_shifted, t, Ts)

# Compute MSE for both reconstructions
MSE_original = np.mean((x_r_t - x_t) ** 2)
MSE_shifted = np.mean((x_r_t_shifted - x_t) ** 2)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x_t, 'k', label='Original Signal')
plt.stem(n, x_n, linefmt='r', markerfmt='ro', basefmt=" ", label='Samples')
plt.plot(t, x_r_t, 'b', linestyle='dashed', label='Reconstructed Signal')
plt.title(f'Reconstruction (MSE = {MSE_original:.2e})')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, x_t, 'k', label='Original Signal')
plt.stem(n_shifted, x_n_shifted, linefmt='g', markerfmt='go', basefmt=" ", label='Shifted Samples')
plt.plot(t, x_r_t_shifted, 'b', linestyle='dashed', label='Reconstructed Signal (Shifted)')
plt.title(f'Shifted Sampling Reconstruction (MSE = {MSE_shifted:.2e})')
plt.legend()

plt.tight_layout()
plt.show()
