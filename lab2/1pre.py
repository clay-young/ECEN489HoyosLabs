import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Fs = 5e6  # Sampling frequency (5 MHz)
F = 2e6   # Signal frequency (2 MHz)
A = 1     # Amplitude (1V)
T = 1e-6  # Duration of the signal (1 microsecond)

# Generate time vector
t = np.arange(0, T, 1/Fs)

# Generate the tone
signal = A * np.sin(2 * np.pi * F * t)

# Plot the signal
plt.figure(figsize=(8,4))
plt.plot(t * 1e6, signal, marker='o', linestyle='-')
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude (V)")
plt.title("2 MHz Tone Sampled at 5 MHz")
plt.grid()
plt.show()
