import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Define FIR and IIR filter coefficients
b_fir = [1, 5, 6]  # FIR filter coefficients
a_fir = [1]        # FIR always has denominator = 1

b_iir = [1, -0.5]  # Example IIR filter numerator
a_iir = [1, -1.5]  # Example IIR filter denominator

# Compute and plot impulse response
N = 50  # Length of impulse response
impulse = np.zeros(N)
impulse[0] = 1  # Delta function

h_fir = signal.lfilter(b_fir, a_fir, impulse)
h_iir = signal.lfilter(b_iir, a_iir, impulse)

plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.stem(h_fir, basefmt=" ")
plt.title("Impulse Response - FIR Filter")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(2,1,2)
plt.stem(h_iir, basefmt=" ")
plt.title("Impulse Response - IIR Filter")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# Compute poles
poles_fir = np.roots(a_fir)  # Always at origin for FIR
poles_iir = np.roots(a_iir)

# Check stability
def check_stability(poles, filter_type):
    stability = all(np.abs(poles) < 1)
    print(f"{filter_type} Filter Stability: {'Stable' if stability else 'Unstable'}")
    return stability

check_stability(poles_fir, "FIR")
check_stability(poles_iir, "IIR")

# Plot pole-zero diagram
plt.figure(figsize=(6,6))
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', linewidth=0.5)

plt.scatter(np.real(poles_fir), np.imag(poles_fir), marker='o', label="FIR Poles", color='blue')
plt.scatter(np.real(poles_iir), np.imag(poles_iir), marker='x', label="IIR Poles", color='red')

plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.legend()
plt.title("Pole Locations")
plt.show()
