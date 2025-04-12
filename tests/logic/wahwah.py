import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal
from scipy.interpolate import interp1d

# signal 
sr = 48000
duration = 5
# 
position = 0.5 
# wah-wah 
min_freq = 400 
max_freq = 2500 
Q_1k = 4  # Q 
LB = -60  # dB
LM = 20   # dB
LT = -60  # dB 
eQ = -1.1 # Q exponent 
eB = 0  # Bass exponent 
eM = -0.1 # Mids exponent 
eT = 0 # Treble exponent 
tau = 0.01 # time constant of parameter smoothing filters (in seconds)
ksmooth = 1 / (tau * sr) # coefficients of the parameter smoothing filters 
pit = 4 * np.atan(1) / sr # frequency scaling factor 

def init_params():
    
    global k, k_u, k_s, kdiv, kdiv_u, kdiv_s, kf, kf_u, kf_s, b0, b0_u, b0_s, kb1, kb1_u, kb1_s, b2, b2_u, b2_s
    global s1l, s2l, s1r, s2r
    # k parameters
    k = 0.1
    k_u = 0.1
    k_s = 0.1
    # kdiv parameters
    kdiv = 1 / 1.11
    kdiv_u = 1 / 1.11
    kdiv_s = 1 / 1.11
    # kf parameters
    kf = 1.1
    kf_u = 1.1
    kf_s = 1.1
    # b0 parameters
    b0 = 0.0
    b0_u = 0.00
    b0_s = 0.0
    # kb1 parameters
    kb1 = 10.0
    kb1_u = 10.0
    kb1_s = 10.0
    # b2 parameters
    b2 = 0.0
    b2_u = 0.00
    b2_s = 0.0

    # filter states 
    s1l = 0.0
    s2l = 0.0
    s1r = 0.0
    s2r = 0.0

def update_coef(position):
    # Parameters of the analog prototype filter
    f = min_freq * np.exp(position * np.log(max_freq/min_freq))
    f_khz = f / 1000
    Q = Q_1k * f_khz ** eQ
    
    # Conditional assignments for filter coefficients
    b0_1k = 0 if LT == -60 else 10 ** (LT * 0.05)
    b1_1k = 0 if LM == -60 else 10 ** (LM * 0.05)
    b2_1k = 0 if LB == -60 else 10 ** (LB * 0.05)
    
    # Update b parameters with frequency scaling
    b0_u = b0_1k * f_khz ** eT
    b1 = b1_1k * f_khz ** eM
    b2_u = b2_1k * f_khz ** eB
    
    # Prewaping of f and Q
    fw = f * np.tan(pit * f) / (pit * f)
    aux = pit * f / np.sin(2 * pit * f) * np.log((np.sqrt(1 + 4 * Q * Q) + 1) / (np.sqrt(1 + 4 * Q * Q) - 1))
    kqw = np.exp(aux) - np.exp(-aux)
    
    # Parameters of the digital state-variable filter
    k_u = pit * fw
    kdiv_u = 1 / (1 + k_u * (k_u + kqw))
    kf_u = kqw + k_u
    kb1_u = b1 * kqw
    
    return k_u, kdiv_u, kf_u, kb1_u, b0_u, b1, b2_u

def wahwah_proc(x_l, x_r):
    global k_s, kdiv_s, kf_s, b0_s, kb1_s, b2_s, s1l, s2l, s1r, s2r
    
    # Parameter smoothing filters
    k = ksmooth * (k_u - k_s) + k_s
    k_s = k
    kdiv = ksmooth * (kdiv_u - kdiv_s) + kdiv_s
    kdiv_s = kdiv
    kf = ksmooth * (kf_u - kf_s) + kf_s
    kf_s = kf
    b0 = ksmooth * (b0_u - b0_s) + b0_s
    b0_s = b0
    kb1 = ksmooth * (kb1_u - kb1_s) + kb1_s
    kb1_s = kb1
    b2 = ksmooth * (b2_u - b2_s) + b2_s
    b2_s = b2
    
    # The state-variable filter - left channel
    hpl = kdiv * (x_l - kf * s1l - s2l)
    aux = k * hpl
    bpl = aux + s1l
    s1l = aux + bpl
    aux = k * bpl
    lpl = aux + s2l
    s2l = aux + lpl
    spl0 = b0 * hpl + kb1 * bpl + b2 * lpl
    
    # The state-variable filter - right channel
    hpr = kdiv * (x_r - kf * s1r - s2r)
    aux = k * hpr
    bpr = aux + s1r
    s1r = aux + bpr
    aux = k * bpr
    lpr = aux + s2r
    s2r = aux + lpr
    spl1 = b0 * hpr + kb1 * bpr + b2 * lpr
    
    return spl0, spl1

def process_audio(input_signal, position=0.5):
    """
    Process a stereo audio signal through the wah-wah effect.
    
    Args:
        input_signal: numpy array of shape (2, samples)
        position: wah-wah pedal position (0 to 1)
    
    Returns:
        processed_signal: numpy array of shape (2, samples), normalized to [-1, 1]
    """
    
    # Initialize output array
    num_samples = input_signal.shape[1]
    output_signal = np.zeros_like(input_signal)
    
    # Update coefficients based on position
    k_u, kdiv_u, kf_u, kb1_u, b0_u, b1, b2_u = update_coef(position)
    
    # Process sample by sample
    for i in range(num_samples):
        # Get input samples for left and right channels
        x_l = input_signal[0, i]
        x_r = input_signal[1, i]
        
        # Process through wah-wah
        spl0, spl1 = wahwah_proc(x_l, x_r)
        
        # Store in output array
        output_signal[0, i] = spl0
        output_signal[1, i] = spl1
    
    # Normalize the output to [-1, 1] range
    max_abs_val = np.max(np.abs(output_signal))
    if max_abs_val > 1.0:
        output_signal = output_signal / max_abs_val
        print(f"Signal was normalized by factor: {1/max_abs_val:.3f}")
    
    return output_signal

def calculate_frequency_response(freqs, position):
    """
    Calculate the frequency response of the wah-wah effect at a given position.
    
    Args:
        freqs: Array of frequencies to analyze
        position: Wah-wah pedal position (0 to 1)
    
    Returns:
        magnitude_response: Array of magnitude responses in dB
    """
    # Get filter coefficients for the analog prototype
    f = min_freq * np.exp(position * np.log(max_freq/min_freq))
    f_khz = f / 1000
    Q = Q_1k * f_khz ** eQ
    
    # Calculate analog filter coefficients
    b0_1k = 0 if LT == -60 else 10 ** (LT * 0.05)
    b1_1k = 0 if LM == -60 else 10 ** (LM * 0.05)
    b2_1k = 0 if LB == -60 else 10 ** (LB * 0.05)
    
    b0_a = b0_1k * f_khz ** eT
    b1_a = b1_1k * f_khz ** eM
    b2_a = b2_1k * f_khz ** eB
    
    # Apply frequency pre-warping like in the digital implementation
    fw = f * np.tan(pit * f) / (pit * f)
    
    # Calculate frequency response with pre-warped frequencies
    # Pre-warp the analysis frequencies too
    freqs_w = freqs * np.tan(np.pi * freqs / sr) / (np.pi * freqs / sr)
    
    # Transfer function H(f) = (b2 + b1*jf/(Q*fw) - b0*f^2/fw^2) / (1 + jf/(Q*fw) - f^2/fw^2)
    numerator = b2_a + b1_a * (1j * freqs_w)/(Q * fw) - b0_a * (freqs_w**2)/(fw**2)
    denominator = 1 + (1j * freqs_w)/(Q * fw) - (freqs_w**2)/(fw**2)
    
    H = numerator / denominator
    
    # Convert to dB
    magnitude_db = 20 * np.log10(np.abs(H))
    return magnitude_db

def measure_frequency_response(freqs, position):
    """
    Measure the actual frequency response of the wah-wah effect using z-transform analysis.
    
    Args:
        freqs: Array of frequencies to analyze
        position: Wah-wah pedal position (0 to 1)
    
    Returns:
        freq_points: Array of actual frequency points from measurement
        magnitude_response: Array of magnitude responses in dB
    """
    # Get the resonant frequency for this position
    f = min_freq * np.exp(position * np.log(max_freq/min_freq))
    f_khz = f / 1000
    Q = Q_1k * f_khz ** eQ
    
    # Pre-warp the resonant frequency
    fw = f * np.tan(pit * f) / (pit * f)
    
    # Get Q warping
    aux = pit * f / np.sin(2 * pit * f) * np.log((np.sqrt(1 + 4 * Q * Q) + 1) / (np.sqrt(1 + 4 * Q * Q) - 1))
    kqw = np.exp(aux) - np.exp(-aux)
    
    # Get coefficients
    k = pit * fw
    kdiv = 1 / (1 + k * (k + kqw))
    kf = kqw + k
    
    # Get mixing coefficients
    b0_1k = 0 if LT == -60 else 10 ** (LT * 0.05)
    b1_1k = 0 if LM == -60 else 10 ** (LM * 0.05)
    b2_1k = 0 if LB == -60 else 10 ** (LB * 0.05)
    
    b0 = b0_1k * f_khz ** eT
    kb1 = b1_1k * f_khz ** eM * kqw
    b2 = b2_1k * f_khz ** eB
    
    # Convert frequencies to normalized frequency
    w = 2 * np.pi * freqs / sr
    z = np.exp(1j * w)
    
    # State-variable filter transfer functions
    # For the integrator: k/(1-z^(-1))
    int1 = k / (1 - z**(-1))
    
    # Calculate transfer functions
    # From the difference equations:
    # hp = kdiv * (x - kf*s1 - s2)
    # bp = aux + s1; s1 = aux + bp  where aux = k*hp
    # lp = aux + s2; s2 = aux + lp  where aux = k*bp
    
    # This means:
    # bp = 2k*hp/(1-z^(-1))
    # lp = 2k*bp/(1-z^(-1))
    
    # Solve the system:
    denom = 1 + kdiv * kf * 2 * k/(1-z**(-1)) + kdiv * (2*k/(1-z**(-1)))**2
    H_hp = kdiv / denom
    H_bp = 2 * k * H_hp / (1-z**(-1))
    H_lp = 2 * k * H_bp / (1-z**(-1))
    
    # Total response
    H = b0 * H_hp + kb1 * H_bp + b2 * H_lp
    
    # Convert to dB
    mag_db = 20 * np.log10(np.abs(H))
    
    return freqs, mag_db

def plot_frequency_response(positions=[0.0], save_path='wahwah_frequency_response.png'):
    """
    Plot both theoretical and measured frequency response of the wah-wah effect at different positions.
    
    Args:
        positions: List of wah-wah pedal positions to analyze
        save_path: Path to save the plot
    """
    # Generate frequency points (logarithmically spaced)
    freqs = np.logspace(1, 4.5, 1000)  # 10 Hz to 31.6 kHz
    
    plt.figure(figsize=(12, 8))
    
    # Colors for different positions
    colors = ['blue', 'green', 'red']
    measured_colors = ['cyan', 'lime', 'magenta']
    
    # Plot frequency response for each position
    for pos, theo_color, meas_color in zip(positions, colors, measured_colors):
        # Theoretical response
        theoretical_db = calculate_frequency_response(freqs, pos)
        plt.semilogx(freqs, theoretical_db, color=theo_color, linestyle='-', label=f'Theoretical Pos {pos:.1f}')
        
        # Measured response
        meas_freqs, measured_db = measure_frequency_response(freqs, pos)
        plt.semilogx(meas_freqs, measured_db, color=meas_color, linestyle='--', label=f'Measured Pos {pos:.1f}')
    
    plt.grid(True, which="both", ls="-", alpha=0.6)
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Gain / dB')
    plt.title('Wah-Wah Frequency Response: Theoretical vs Measured')
    plt.legend()
    plt.ylim(-40, 25)
    plt.xlim(10, 24000)
    
    # Add vertical lines at characteristic frequencies
    for freq in [400, 1000, 2500]:
        plt.axvline(freq, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Plot both theoretical and measured responses
    plot_frequency_response()
    
    # # Original test signal processing code...
    # duration = 0.001  # seconds
    # t = np.linspace(0, duration, int(sr * duration))
    # # Example: 440 Hz sine wave in left channel, 880 Hz in right channel
    # test_signal = np.array([
    #     np.sin(2 * np.pi * 440 * t),  # left channel
    #     np.sin(2 * np.pi * 880 * t)   # right channel
    # ])
    
    # init_params()
    # # Process the signal
    # processed_signal = process_audio(test_signal, position=0.5)
    
    # # Plot the results
    # plt.figure(figsize=(15, 5))
    # plt.subplot(2, 1, 1)
    # plt.plot(t, test_signal[0], label='Original Left')
    # plt.plot(t, processed_signal[0], label='Processed Left')
    # plt.legend()
    # plt.title('Left Channel')
    
    # plt.subplot(2, 1, 2)
    # plt.plot(t, test_signal[1], label='Original Right')
    # plt.plot(t, processed_signal[1], label='Processed Right')
    # plt.legend()
    # plt.title('Right Channel')
    # plt.tight_layout()
    
    # # Save the plot instead of showing it
    # plt.savefig('wahwah_effect_comparison.png', dpi=300, bbox_inches='tight')
    # plt.close()  # Close the figure to free memory



