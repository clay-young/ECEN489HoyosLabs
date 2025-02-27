import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, windows

def quantize(signal, num_bits):
    levels = 2**num_bits
    delta = 2 / levels  # Assuming full-scale range [-1, 1]
    return np.round(signal / delta) * delta

# Parameters
fs = 400e6  # Sampling Frequency (400 MHz)
f_in = 200e6  # Input sine wave frequency (200 MHz)
num_periods = 100

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

def plot_psd(signal, fs, title, use_hanning=False):
    nperseg = min(1024, len(signal))  # Adjust nperseg dynamically
    window = windows.hann(nperseg) if use_hanning else 'boxcar'
    f, Pxx = welch(signal, fs, window=window, nperseg=nperseg)
    plt.figure()
    plt.semilogy(f, Pxx)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(title)
    plt.grid()
    plt.show()

# Generate Signal
t, signal = generate_signal(f_in, fs, num_periods)

# Test for N = 6 and N = 12 with Hanning window
for num_bits in [6, 12]:
    quantized = quantize(signal, num_bits)
    snr = compute_snr(signal, quantized)
    print(f"SNR for {num_bits}-bit quantizer with Hanning window: {snr:.2f} dB (Expected: {6*num_bits} dB)")
    plot_psd(quantized, fs, f"PSD of {num_bits}-bit Quantized Signal with Hanning Window", use_hanning=True)
