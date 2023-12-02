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
nyquist_freq = Music_Sample_Rate / 2
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
