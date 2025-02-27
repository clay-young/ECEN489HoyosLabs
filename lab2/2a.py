import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def quantize(signal, num_bits):
    levels = 2**num_bits
    delta = 2 / levels  # Assuming full-scale range [-1, 1]
    return np.round(signal / delta) * delta

# Parameters
fs = 417e6  # Sampling frequency (400 MHz) Can be changed to sample at incommensurate frequencies
f_in = 200e6  # Input sine wave frequency (200 MHz)
num_bits = 6  # 6-bit quantizer
num_periods_30 = 30
num_periods_100 = 100

def generate_signal(f_in, fs, num_periods):
    T = 1 / f_in  # Period of sine wave
    t = np.arange(0, num_periods * T, 1 / fs)
    signal = np.sin(2 * np.pi * f_in * t)
    return t, signal / np.max(np.abs(signal))  # Normalize signal

def compute_snr(original, quantized):
    noise = quantized - original
    P_signal = np.mean(original**2)
    P_noise = np.mean(noise**2)
    return 10 * np.log10(P_signal / P_noise) if P_noise > 0 else float('inf')

def plot_psd(signal, fs, title):
    nperseg = min(1024, len(signal))  # Adjust nperseg dynamically
    f, Pxx = welch(signal, fs, nperseg=nperseg)
    plt.figure()
    plt.semilogy(f, Pxx)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(title)
    plt.grid()
    plt.show()

# 30 Periods
t_30, signal_30 = generate_signal(f_in, fs, num_periods_30)
quantized_30 = quantize(signal_30, num_bits)
snr_30 = compute_snr(signal_30, quantized_30)
print(f"SNR for 30 periods: {snr_30:.2f} dB")
plot_psd(quantized_30, fs, "PSD of Quantized Signal (30 Periods)")

# 100 Periods
t_100, signal_100 = generate_signal(f_in, fs, num_periods_100)
quantized_100 = quantize(signal_100, num_bits)
snr_100 = compute_snr(signal_100, quantized_100)
print(f"SNR for 100 periods: {snr_100:.2f} dB")
plot_psd(quantized_100, fs, "PSD of Quantized Signal (100 Periods)")
