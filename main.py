from IPython.display import Audio
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import butter, lfilter

Noise_Wav = wavfile.read("Noise.wav")
Music_Wav = wavfile.read("Music.wav")

# Separate the object elements for Music
Music_Sample_Rate = Music_Wav[0]  # Sample Rate
Music_Audio = Music_Wav[1][:, 0]  # Signal Data
Music_Length = Music_Audio.size  # Number of samples
Music_Sample_Period = 1 / Music_Sample_Rate  # Sample Period Music_Sample_Period
Music_Time_Vector = np.arange(0, Music_Length) * \
    Music_Sample_Period  # Time vector for Music

# Separate the object elements for Noise
Noise_Sample_Rate = Noise_Wav[0]  # Sample Rate
Noise_Audio = Noise_Wav[1][:, 0]  # Signal Data
Noise_Length = Noise_Audio.size  # Number of samples
Noise_Sample_Period = 1 / Noise_Sample_Rate  # Sample Period Noise_Sample_Period
Noise_Time_Vector = np.arange(0, Noise_Length) * \
    Noise_Sample_Period  # Time vector for Noise

# Ensure both signals have the same sample rate (Fs)
if Music_Sample_Rate != Noise_Sample_Rate:
    raise ValueError("Sample rates of the two audio files must match.")

# Check the durations of the two audio files and determine the shorter duration
min_duration = min(Music_Length, Noise_Length)

# Truncate both signals to the common length
Music_Audio = Music_Audio[:min_duration]
Noise_Audio = Noise_Audio[:min_duration]

# Combine the two audio files
Combined_audio = Music_Audio + Noise_Audio

# Create a time vector based on the common length
Combined_Time_Vector = np.arange(0, min_duration) / Music_Sample_Rate

# Play the individual audio files and the combined audio
print("Playing Music...")
Audio(Music_Audio, rate=Music_Sample_Rate, autoplay=True)

print("Playing Noise...")
Audio(Noise_Audio, rate=Noise_Sample_Rate, autoplay=True)

print("Playing Combined Audio...")
Audio(Combined_audio, rate=Music_Sample_Rate, autoplay=True)

# Plot the combined audio waveform
plt.figure(figsize=(10, 6))
plt.plot(Combined_Time_Vector, Combined_audio, color='b',
         linewidth=1.5, label='Combined Audio')
plt.xlim(Combined_Time_Vector[0], Combined_Time_Vector[-1])
plt.title('Combined Audio Waveform')
plt.ylabel('Amplitude')
plt.xlabel('Time (sec.)')
plt.legend()
plt.show()

# Calculate the FFT of the combined audio
fft_combined = fft(Combined_audio)

# Define the noise frequency range for the band-stop filter
noise_min_freq = 250  # Example: minimum noise frequency in Hz
noise_max_freq = 700  # Example: maximum noise frequency in Hz

# Design a band-stop filter in the frequency domain to attenuate the noise
order = 4  # Example filter order
nyquist_freq = Music_Sample_Rate / 2
lowcut = noise_min_freq / nyquist_freq
highcut = noise_max_freq / nyquist_freq


b, a = butter(order, [lowcut, highcut], btype='bandstop')


# Apply the band-stop filter in the frequency domain
filtered_audio_bandstop = ifft(
    fft_combined * (1 - lfilter(b, a, np.ones_like(fft_combined)))).real


# Create a time vector for the cleaned audio
t_cleaned_bandstop = np.arange(0, min_duration) / Music_Sample_Rate

# Plot the cleaned audio waveform after the band-stop filter
plt.figure(figsize=(10, 6))
plt.plot(t_cleaned_bandstop, filtered_audio_bandstop, color='g',
         linewidth=1.5, label='Cleaned Audio (Band-Stop)')
plt.xlim(t_cleaned_bandstop[0], t_cleaned_bandstop[-1])
plt.title('Cleaned Audio Waveform (Band-Stop Filter)')
plt.ylabel('Amplitude')
plt.xlabel('Time (sec.)')
plt.legend()
plt.show()

# Calculate the FFT of the cleaned audio after the band-stop filter
fft_cleaned_bandstop = fft(filtered_audio_bandstop)
freq_cleaned_bandstop = np.fft.fftfreq(
    len(fft_cleaned_bandstop), 1 / Music_Sample_Rate)

# Plot the frequency spectrum after the band-stop filter
plt.figure(figsize=(10, 6))
plt.plot(freq_cleaned_bandstop, np.abs(fft_cleaned_bandstop),
         color='b', linewidth=1.5, label='Frequency Spectrum (Band-Stop)')
plt.xlim(0, 1000)  # Display up to the Nyquist frequency
plt.title('Frequency Spectrum (Band-Stop Filter)')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.legend()
plt.tight_layout()
plt.show()

# Play the cleaned audio
print("Playing Cleaned Audio...")
Audio(filtered_audio_bandstop, rate=Music_Sample_Rate, autoplay=True)

# Design a high-pass filter for noise reduction
cutoff_freq_highpass = 300  # Example high-pass cutoff frequency in Hz
order_highpass = 4  # Example filter order
highcut_highpass = cutoff_freq_highpass / nyquist_freq

# Create the high-pass filter
b_highpass, a_highpass = butter(order_highpass, highcut_highpass, btype='high')

# Apply the high-pass filter to the combined audio
filtered_audio_highpass = lfilter(b_highpass, a_highpass, Combined_audio)

# Create a time vector for the cleaned audio
t_cleaned_highpass = np.arange(0, min_duration) / Music_Sample_Rate

# Plot the cleaned audio waveform after the high-pass filter
plt.figure(figsize=(10, 6))
plt.plot(t_cleaned_highpass, filtered_audio_highpass, color='g',
         linewidth=1.5, label='Cleaned Audio (High-Pass)')
plt.xlim(t_cleaned_highpass[0], t_cleaned_highpass[-1])
plt.title('Cleaned Audio Waveform (High-Pass Filter)')
plt.ylabel('Amplitude')
plt.xlabel('Time (sec.)')
plt.legend()
plt.show()

# Play the cleaned audio
print("Playing Cleaned Audio...")
Audio(filtered_audio_highpass, rate=Music_Sample_Rate, autoplay=True)

# Design a low-pass filter for noise reduction
cutoff_freq_lowpass = 500  # Example low-pass cutoff frequency in Hz
order_lowpass = 2  # Example filter order
lowcut_lowpass = cutoff_freq_lowpass / nyquist_freq

# Create the low-pass filter
b_lowpass, a_lowpass = butter(order_lowpass, lowcut_lowpass, btype='low')


# Apply the low-pass filter to the combined audio
filtered_audio_lowpass = lfilter(b_lowpass, a_lowpass, Combined_audio)

# Create a time vector for the cleaned audio
t_cleaned_lowpass = np.arange(0, min_duration) / Music_Sample_Rate

# Plot the cleaned audio waveform after the low-pass filter
plt.figure(figsize=(10, 6))
plt.plot(t_cleaned_lowpass, filtered_audio_lowpass, color='g',
         linewidth=1.5, label='Cleaned Audio (Low-Pass)')
plt.xlim(t_cleaned_lowpass[0], t_cleaned_lowpass[-1])
plt.title('Cleaned Audio Waveform (Low-Pass Filter)')
plt.ylabel('Amplitude')
plt.xlabel('Time (sec.)')
plt.legend()
plt.show()

# Play the cleaned audio
print("Playing Cleaned Audio...")
Audio(filtered_audio_lowpass, rate=Music_Sample_Rate, autoplay=True)

# Define the desired frequency band for the bandpass filter
bandpass_min_freq = 200  # Example: lower frequency in Hz
bandpass_max_freq = 800  # Example: upper frequency in Hz

# Design a bandpass filter
order_bandpass = 4  # Example filter order
lowcut_bandpass = bandpass_min_freq / nyquist_freq
highcut_bandpass = bandpass_max_freq / nyquist_freq

# Create the bandpass filter
b_bandpass, a_bandpass = butter(
    order_bandpass, [lowcut_bandpass, highcut_bandpass], btype='band')

# Apply the bandpass filter to the combined audio
filtered_audio_bandpass = lfilter(b_bandpass, a_bandpass, Combined_audio)

# Create a time vector for the cleaned audio
t_cleaned_bandpass = np.arange(0, min_duration) / Music_Sample_Rate

# Plot the cleaned audio waveform after the bandpass filter
plt.figure(figsize=(10, 6))
plt.plot(t_cleaned_bandpass, filtered_audio_bandpass, color='g',
         linewidth=1.5, label='Cleaned Audio (Band-Pass)')
plt.xlim(t_cleaned_bandpass[0], t_cleaned_bandpass[-1])
plt.title('Cleaned Audio Waveform (Band-Pass Filter)')
plt.ylabel('Amplitude')
plt.xlabel('Time (sec.)')
plt.legend()
plt.show()

# Play the cleaned audio
print("Playing Cleaned Audio...")
Audio(filtered_audio_bandpass, rate=Music_Sample_Rate, autoplay=True)

# Define the step size and filter length for the LMS adaptive filter
step_size_lms = 0.000001  # Adjust as needed
filter_length_lms = 64  # Adjust as needed


def lms_adaptive_filter(input_signal, noise_signal, step_size, filter_length):
    # Normalize the input signals
    input_signal = input_signal / np.max(np.abs(input_signal))
    noise_signal = noise_signal / np.max(np.abs(noise_signal))

    # Determine the common length for both signals
    min_duration = min(len(input_signal), len(noise_signal))

    # Truncate both signals to the common length
    input_signal = input_signal[:min_duration]
    noise_signal = noise_signal[:min_duration]

    # Initialize the adaptive filter coefficients
    filter_coeffs = np.zeros(filter_length)

    # Initialize the filtered signal
    filtered_signal = np.zeros_like(input_signal)

    # Iterate over the input signal
    for n in range(min_duration):
        # Determine the valid indices for the input window
        start_idx = max(0, n - filter_length + 1)
        end_idx = n + 1

        # Extract the input window and pad with zeros if necessary
        input_window = input_signal[start_idx:end_idx]
        if len(input_window) < filter_length:
            input_window = np.concatenate(
                ([0] * (filter_length - len(input_window)), input_window))

        # Compute the filter output
        filter_output = np.dot(filter_coeffs, input_window)

        # Compute the error (difference between noise and filter output)
        error = noise_signal[n] - filter_output

        # Update the filter coefficients
        filter_coeffs += 2 * step_size * error * input_window

        # Apply the filter to the input signal
        filtered_signal[n] = filter_output

    return filtered_signal


# Apply the LMS adaptive filter to remove noise
filtered_audio_lms = lms_adaptive_filter(
    Combined_audio, Noise_Audio, step_size_lms, filter_length_lms)

# Play the cleaned audio after LMS adaptive filter
print("Playing Cleaned Audio (LMS Adaptive Filter)...")
Audio(filtered_audio_lms, rate=Music_Sample_Rate, autoplay=True)
