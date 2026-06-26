"""
fingerprint.py — Core Lib
"""

import numpy as np
import librosa
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure

# Defaul
DEFAULT_SR = 22050
DEFAULT_N_FFT = 1024
DEFAULT_HOP = 512
PEAK_NEIGHBORHOOD_SIZE = 10  
AMP_MIN_DB = -55             
FAN_OUT = 15                         
MIN_TIME_DELTA = 1                  
MAX_TIME_DELTA = 50                 
MAX_FREQ_BIN = 512                   


# Audio Load 
def load_audio(filepath, sr=DEFAULT_SR):
    """Load an audio file, convert to mono, resample to *sr*."""
    y, sr_out = librosa.load(filepath, sr=sr, mono=True)
    return y, sr_out



def compute_spectrogram(y, sr=DEFAULT_SR, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP):
    """
    Compute the magnitude STFT spectrogram in dB.

    Returns
    -------
    S_db   : ndarray (n_freq, n_time)   — magnitude in dB
    freqs  : ndarray (n_freq,)          — frequency axis (Hz)
    times  : ndarray (n_time,)          — time axis (seconds)
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    return S_db, freqs, times


def find_peaks(S_db, neighborhood_size=PEAK_NEIGHBORHOOD_SIZE, amp_min=AMP_MIN_DB):
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, neighborhood_size)

    local_max = maximum_filter(S_db, footprint=neighborhood) == S_db
    detected = local_max & (S_db > amp_min)

    # Extract (freq_bin, time_frame) coordinates
    peaks = np.argwhere(detected)
    return peaks


#  Fingerprint Has
def generate_hashes(peaks, fan_out=FAN_OUT,
                    min_dt=MIN_TIME_DELTA, max_dt=MAX_TIME_DELTA):
   
    idx = np.argsort(peaks[:, 1])
    peaks_sorted = peaks[idx]

    hashes = []
    n = len(peaks_sorted)
    for i in range(n):
        f1, t1 = int(peaks_sorted[i, 0]), int(peaks_sorted[i, 1])
        count = 0
        for j in range(i + 1, n):
            f2, t2 = int(peaks_sorted[j, 0]), int(peaks_sorted[j, 1])
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            hashes.append(((f1, f2, dt), t1))
            count += 1
            if count >= fan_out:
                break
    return hashes


def generate_single_peak_hashes(peaks):
    hashes = []
    for freq_bin, time_frame in peaks:
        hashes.append(((int(freq_bin),), int(time_frame)))
    return hashes


# Matchin
def match_hashes(query_hashes, database):
 
    offset_counts = {}

    for h, q_time in query_hashes:
        if h in database:
            for song_id, db_time in database[h]:
                offset = db_time - q_time
                if song_id not in offset_counts:
                    offset_counts[song_id] = {}
                oc = offset_counts[song_id]
                oc[offset] = oc.get(offset, 0) + 1

    best_song_id = None
    best_count = 0
    for song_id, offsets in offset_counts.items():
        max_count = max(offsets.values()) if offsets else 0
        if max_count > best_count:
            best_count = max_count
            best_song_id = song_id

    return best_song_id, best_count, offset_counts


def get_offset_histogram(offset_counts_for_song):
    """Convert offset dict → sorted arrays for plotting."""
    if not offset_counts_for_song:
        return np.array([]), np.array([])
    offsets = sorted(offset_counts_for_song.keys())
    counts = [offset_counts_for_song[o] for o in offsets]
    return np.array(offsets), np.array(counts)


# Audio Manipulate
def add_noise(y, snr_db=10):
    """Add white Gaussian noise at the specified SNR (dB)."""
    rms_signal = np.sqrt(np.mean(y ** 2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, len(y))
    return y + noise


def pitch_shift(y, sr=DEFAULT_SR, n_steps=1):
    """Shift pitch by *n_steps* semitones."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def time_stretch(y, rate=1.0):
    """Time-stretch the audio by *rate* (>1 = faster, <1 = slower)."""
    return librosa.effects.time_stretch(y=y, rate=rate)
