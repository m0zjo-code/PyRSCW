import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
import wavio

def open_wav_file(filename):
    wav_data = wavio.read(filename)
    print("Wavfile loaded. Len:%i, Fs:%iHz" % (wav_data.data.shape[0], wav_data.rate))
    wav_data.data = wav_data.data[:,0]
    return wav_data

def plot_file(wav_data):
    plt.plot(wav_data.data[0:1000])
    plt.show()
    return
    
def remove_dc(wav_data):
    wav_data.data = wav_data.data - np.mean(wav_data.data)
    return wav_data

def find_carrier(wav_data):
    f, Pxx_den = signal.welch(wav_data.data, wav_data.rate, nperseg=2048)
    
    max_value = np.argmax(Pxx_den)
    
    print("Carrier found at %0.2fHz" % f[max_value])
    
    carrier_freq = f[max_value]
    
    plot = False
    if plot:
        plt.semilogy(f, Pxx_den)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()
    
    return carrier_freq

def generate_carriers(wav_data, carrier_freq):
    rate = wav_data.rate  # samples per second
    sample_length = len(wav_data.data)         # sample duration (seconds)
    f = carrier_freq     # sound frequency (Hz)
    t = np.linspace(0, sample_length/rate, sample_length, endpoint=False)
    sin_car = np.sin(2*np.pi * f * t)
    cos_car = np.cos(2*np.pi * f * t)
    print("Carriers Generated")
    return sin_car, cos_car

def design_lpf(fs, mode, plot = False):
    
    nyq_rate = fs/2
    
    
    if mode == 0:
        with open("150hzlpf", "r") as f:
            taps = []
            for tap in f:
                taps.append(float(tap))
                
        print("Filter of order %i loaded" % len(taps))
    
    if mode == 1:
        # Help from -->> https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.  We'll design the filter
        # with a 5 Hz transition width.
        width = 500/nyq_rate

        # The desired attenuation in the stop band, in dB.
        ripple_db = 40.0

        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = signal.kaiserord(ripple_db, width)

        # The cutoff frequency of the filter.
        cutoff_hz = 100.0

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = signal.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
        
        print("Filter of order %i deisgned" % len(taps))
    
    if plot:
        w, h = signal.freqz(taps, worN=8000)
        plt.plot((w/np.pi)*nyq_rate, 20 * np.log(np.absolute(h)), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response')
        #ylim(-0.05, 1.05)
        plt.grid(True)
        plt.show()
        
    return taps

def generate_filtered_baseband(wav_data, carriers):
    
    taps = design_lpf(wav_data.rate, 1)
    
    i_wave = carriers[0] * wav_data.data
    q_wave = carriers[1] * wav_data.data
    
    # Use lfilter to filter x with the FIR filter.
    filtered_i = signal.lfilter(taps, 1.0, i_wave)
    filtered_q = signal.lfilter(taps, 1.0, q_wave)
    
    return filtered_i, filtered_q

def downsample_abs_bb(baseband_data, fs):
    i_chan = baseband_data[0]
    q_chan = baseband_data[1]
    
    N = 48 # 48000/48 = 1000
    bb_len = len(i_chan)
    
    truncate_len = np.mod(bb_len, N)
    i_chan = i_chan[:bb_len-truncate_len]
    q_chan = q_chan[:bb_len-truncate_len]
    
    i_chan_ds = i_chan.reshape((int(len(i_chan)/N), N))
    q_chan_ds = q_chan.reshape((int(len(q_chan)/N), N))
    
    i_chan_ds = np.sum(i_chan_ds, axis = 1)
    q_chan_ds = np.sum(q_chan_ds, axis = 1)
    
    return i_chan_ds, q_chan_ds
    
def compute_abs(downsampled_power_data):
    
    sqr_dat = np.square(downsampled_power_data[0]) + np.square(downsampled_power_data[1])
    
    return np.sqrt(sqr_dat)

def smooth_mag(mag, window_len = 10, No_iter = 20):
    for i in range(0, No_iter):
        mag = np.convolve(mag, np.ones((window_len,))/window_len, mode='valid')
    return mag

def get_threshold_val(mag, location, thresh = 0.5, search_len = 1000):
    
    min_search = location-search_len/2
    
    if min_search < 0:
        min_search = 0
        
    max_search = location+search_len/2
    
    if max_search > len(mag) - 1:
        max_search = len(mag) - 1

    search_vector = mag[int(min_search):int(max_search)]
    
    min_val = np.min(search_vector)
    max_val = np.max(search_vector)
    
    delta = max_val - min_val
    
    threshold_val = min_val + delta*thresh
    
    return threshold_val

def get_threshold(mag):
    
    thresh_vector = np.zeros(mag.shape)
    for i in range(0, len(mag)):
        thresh_vector[i] = get_threshold_val(mag, i)
    
    return thresh_vector

def signal_detect_val(mag, location, thresh = 0.5, search_len = 1000):
    
    min_search = location-search_len/2
    
    if min_search < 0:
        min_search = 0
        
    max_search = location+search_len/2
    
    if max_search > len(mag) - 1:
        max_search = len(mag) - 1

    search_vector = mag[int(min_search):int(max_search)]
    
    return np.std(search_vector)
    
def signal_detect(mag, detection_threshold = 50000):
    
    signal_detect_vector = np.zeros(mag.shape)
    for i in range(0, len(mag)):
        signal_detect_vector[i] = signal_detect_val(mag, i)
    
    signal_detect_vector = (signal_detect_vector > detection_threshold)
    
    return signal_detect_vector

def apply_threshold(mag, thresh_vector):
    return mag - thresh_vector

def quantise(mag):
    
    bitstream_ind_pos = mag>0
    bitstream_ind_neg = mag<=0
    
    mag[bitstream_ind_pos] = 1
    mag[bitstream_ind_neg] = -1
    
    return mag

def bit_synch(bitstream, signal_present, min_length = 1000):
    # Finds the start of each bitstream when a signal is present
    signal_len = len(bitstream)
    
    SIGNAL_STATUS = False
    SYNCH = False
    
    signal_list = []
    
    for i in range(1, signal_len):
        if (signal_present[i] == 1) and (signal_present[i-1] == 0):
            #Start of signal
            SIGNAL_STATUS = True
        if (bitstream[i] == 1) and (bitstream[i-1] == -1) and (SIGNAL_STATUS == True) and (SYNCH == False):
            #Start of signal
            #print("Found Signal at %i" % i)
            start_pos = i
            SYNCH = True
        if (signal_present[i] == 0) and (signal_present[i-1] == 1) and (SYNCH == True):
            #End of signal
            #print("Signal End at %i" % i)
            end_pos = i
            signal_list.append([start_pos, end_pos])
            SIGNAL_STATUS = False
            SYNCH = False
        
    if SYNCH == True:
        end_pos = signal_len - 1
        signal_list.append([start_pos, end_pos])
    
    return signal_list
                    
                
def generate_alphabet():
    return
        
        

def plot_mag_data(mag):
    
    for i in mag:
        plt.plot(i)
    plt.show()
    return
