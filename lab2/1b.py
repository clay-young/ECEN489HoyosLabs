import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, windows

# Signal parameters
Fs = 5e6  # Sampling frequency
F = 2e6   # Signal frequency
A = 1.0   # Amplitude
N = 2048  # Number of samples

# Generate time-domain signal
t = np.arange(N) / Fs
signal = A * np.sin(2 * np.pi * F * t)

# Add Gaussian noise for SNR = 50 dB
SNR_dB = 50
signal_power = (A**2) / 2  # Power of sinewave
noise_variance = signal_power / (10**(SNR_dB / 10))
noise = np.random.normal(0, np.sqrt(noise_variance), N)
noisy_signal = signal + noise

# Define windows
windows_dict = {
    "Hanning": windows.hann(N),
    "Hamming": windows.hamming(N),
    "Blackman": windows.blackman(N)
}

for name, window in windows_dict.items():
    # Apply window
    windowed_signal = noisy_signal * window
    
    # Compute DFT (Power Spectral Density)
    f, Pxx = welch(windowed_signal, Fs, nperseg=N, scaling="spectrum")
    
    # Correct signal power for window loss
    window_correction = np.sum(window**2) / N
    Pxx /= window_correction  # Normalize PSD

    # Compute SNR from DFT
    signal_power_DFT = np.max(Pxx)
    noise_indices = (f < (F - 100000)) | (f > (F + 100000))
    noise_power_DFT = np.mean(Pxx[noise_indices])

    SNR_DFT_dB = 10 * np.log10(signal_power_DFT / noise_power_DFT)

    print(f"{name} Window - Estimated SNR: {SNR_DFT_dB:.2f} dB")

    # Plot PSD
    plt.figure(figsize=(10, 5))
    plt.semilogy(f, Pxx)
    plt.title(f"Power Spectral Density ({name} Window)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (V^2/Hz)")
    plt.grid()
    plt.show()
