import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

# Define parameters
F = 2e6      # Frequency of the signal (2 MHz)
Fs = 5e6     # Sampling frequency (5 MHz)
N = 50       # Number of points for DFT
T = 1 / Fs   # Sampling period

# Generate time samples
t = np.arange(N) * T  # N samples spaced by T

# Generate the signal
x_t = np.cos(2 * np.pi * F * t)

# Compute the DFT using FFT
X_f = fft(x_t)

# Frequency axis
frequencies = np.fft.fftfreq(N, d=T)  # Compute frequency bins

# Plot the magnitude spectrum
plt.figure(figsize=(8, 4))
plt.stem(frequencies[:N//2], np.abs(X_f[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum of the 50-point DFT")
plt.grid()
plt.show()

# Define given parameters
F1 = 200e6  # 200 MHz
F2 = 400e6  # 400 MHz
Fs = 1e9    # Sampling frequency (1 GHz)
N = 50      # Number of DFT points

# Generate time samples
t = np.arange(N) / Fs  # N samples spaced by 1/Fs

# Define the signal y(t)
y_t = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F2 * t)

# Compute the 50-point DFT
Y_f = fft(y_t, N)

# Compute frequency bins
frequencies = fftfreq(N, d=1/Fs)

# Plot the magnitude spectrum
plt.figure(figsize=(8, 4))
plt.stem(frequencies[:N//2], np.abs(Y_f[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum of y(t)")
plt.grid()
plt.show()

# New sampling frequency
Fs_new = 500e6  # 500 MHz

# Generate new time samples
t_new = np.arange(N) / Fs_new

# Define the new sampled signal
y_t_new = np.cos(2 * np.pi * F1 * t_new) + np.cos(2 * np.pi * F2 * t_new)

# Compute the 50-point DFT
Y_f_new = fft(y_t_new, N)

# Compute frequency bins
frequencies_new = fftfreq(N, d=1/Fs_new)

# Plot the magnitude spectrum
plt.figure(figsize=(8, 4))
plt.stem(frequencies_new[:N//2], np.abs(Y_f_new[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum with Fs = 500 MHz")
plt.grid()
plt.show()


from numpy import blackman

# Apply Blackman window to x(t) and y(t)
blackman_window = blackman(N)

# Windowed signals

x_t_windowed = x_t * blackman_window #apply to x(t)
y_t_windowed = y_t * blackman_window  # Apply to y(t)
y_t_new_windowed = y_t_new * blackman_window  # Apply to y(t) with Fs = 500MHz

# Compute DFT after windowing
X_f_windowed = fft(x_t_windowed, N)
Y_f_windowed = fft(y_t_windowed, N)
Y_f_new_windowed = fft(y_t_new_windowed, N)

# Plot results with Blackman window (Fs = 5MHz)
plt.figure(figsize=(8, 4))
plt.stem(frequencies[:N//2], np.abs(X_f_windowed[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum of x(t) with Blackman Window (Fs = 5MHz)")
plt.grid()
plt.show()

# Plot results with Blackman window (Fs = 1GHz)
plt.figure(figsize=(8, 4))
plt.stem(frequencies[:N//2], np.abs(Y_f_windowed[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum y(t) with Blackman Window (Fs = 1GHz)")
plt.grid()
plt.show()

# Plot results with Blackman window (Fs = 500MHz)
plt.figure(figsize=(8, 4))
plt.stem(frequencies_new[:N//2], np.abs(Y_f_new_windowed[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum y(t) with Blackman Window (Fs = 500MHz)")
plt.grid()
plt.show()
