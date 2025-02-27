import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Define parameters
Fs = 5e6  # Sampling frequency (5 MHz)
F = 2e6   # Signal frequency (2 MHz)
A = 1     # Amplitude (1V)
T = 1e-6  # Duration of the signal (1 microsecond)
SNR_dB = 50  # Target SNR in dB

# Generate time vector
t = np.arange(0, T, 1/Fs)

# Generate the clean sinewave signal
signal = A * np.sin(2 * np.pi * F * t)

# Calculate signal power
signal_power = np.mean(signal**2)

# Calculate noise variance for target SNR
SNR_linear = 10**(SNR_dB / 10)  # Convert dB to linear scale
noise_variance = signal_power / SNR_linear

# Generate Gaussian noise
noise = np.random.normal(0, np.sqrt(noise_variance), size=t.shape)

# Add noise to the signal
noisy_signal = signal + noise

# Compute PSD using Welch's method
f, Pxx = welch(noisy_signal, Fs, nperseg=256)

# Plot the noisy signal
plt.figure(figsize=(8,4))
plt.plot(t * 1e6, noisy_signal, marker='o', linestyle='-', alpha=0.7, label="Noisy Signal")
plt.plot(t * 1e6, signal, linestyle='-', alpha=0.5, label="Original Signal")
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude (V)")
plt.title("Noisy 2 MHz Tone (SNR = 50 dB)")
plt.legend()
plt.grid()
plt.show()

# Plot PSD
plt.figure(figsize=(8,4))
plt.semilogy(f/1e6, Pxx)  # Convert Hz to MHz for x-axis
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power Spectral Density")
plt.title("Power Spectral Density of Noisy Signal")
plt.grid()
plt.show()

# Compute SNR from PSD
signal_power_DFT = np.max(Pxx)  # Power of main tone
noise_power_DFT = np.mean(Pxx[f > (F + 50000)])  # Estimate noise power from off-band
SNR_DFT_dB = 10 * np.log10(signal_power_DFT / noise_power_DFT)

print(f"Theoretical SNR: {SNR_dB} dB")
print(f"Estimated SNR from DFT: {SNR_DFT_dB:.2f} dB")

# Calculate variance for uniform noise
uniform_variance = 3 * noise_variance  # Uniform noise variance is 3 times that of Gaussian noise
print(f"Variance of Gaussian noise: {noise_variance:.2e}")
print(f"Variance of uniform noise to achieve same SNR: {uniform_variance:.2e}")

# Compute SNR from PSD
signal_power_DFT = np.max(Pxx)  # Power at the main tone

# Estimate noise power by averaging PSD values away from the main frequency
noise_indices = (f < (F - 100000)) | (f > (F + 100000))  # Avoid 2 MHz ± 100 kHz
noise_power_DFT = np.mean(Pxx[noise_indices])  # Estimate noise power

# Ensure noise power isn't zero
if noise_power_DFT > 0:
    SNR_DFT_dB = 10 * np.log10(signal_power_DFT / noise_power_DFT)
else:
    SNR_DFT_dB = float('inf')  # Assign infinity if noise is too small

print(f"Theoretical SNR: {SNR_dB} dB")
print(f"Estimated SNR from DFT: {SNR_DFT_dB:.2f} dB")
