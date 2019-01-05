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
            
            #print(end_pos - start_pos)    
            if (end_pos - start_pos) > min_length:
                signal_list.append([start_pos, end_pos])
            SIGNAL_STATUS = False
            SYNCH = False
        
    if SYNCH == True:
        end_pos = signal_len - 1
        #print(end_pos - start_pos)
        if (end_pos - start_pos) > min_length:
            signal_list.append([start_pos, end_pos])
        
    
    return signal_list
             
# Decoder...

                
def generate_alphabet():
    
    alphabet = {}
    alphabet["a"] = "10111000"
    alphabet["b"] = "111010101000"
    alphabet["c"] = "11101011101000"
    alphabet["d"] = "1110101000"
    alphabet["e"] = "1000"
    alphabet["f"] = "101011101000"
    alphabet["g"] = "111011101000"
    alphabet["h"] = "1010101000"
    alphabet["i"] = "101000"
    alphabet["j"] = "1011101110111000"
    alphabet["k"] = "111010111000"
    alphabet["l"] = "101110101000"
    alphabet["m"] = "1110111000"
    alphabet["n"] = "11101000"
    alphabet["o"] = "11101110111000"
    alphabet["p"] = "10111011101000"
    alphabet["q"] = "1110111010111000"
    alphabet["r"] = "1011101000"
    alphabet["s"] = "10101000"
    alphabet["t"] = "111000"
    alphabet["u"] = "1010111000"
    alphabet["v"] = "101010111000"
    alphabet["w"] = "101110111000"
    alphabet["x"] = "11101010111000"
    alphabet["y"] = "1110101110111000"
    alphabet["z"] = "11101110101000"
    alphabet["0"] = "1110111011101110111000"
    alphabet["1"] = "10111011101110111000"
    alphabet["2"] = "101011101110111000"
    alphabet["3"] = "1010101110111000"
    alphabet["4"] = "10101010111000"
    alphabet["5"] = "101010101000"
    alphabet["6"] = "11101010101000"
    alphabet["7"] = "1110111010101000"
    alphabet["8"] = "111011101110101000"
    alphabet["9"] = "11101110111011101000"
    alphabet[" "] = "0000"
    
    return alphabet
        
def wpm_to_baud_rate(wpm):
    # From -->> https://en.wikipedia.org/wiki/Words_per_minute
    return float((50/60) * wpm)

#def upsample_alphabet(alphabet, baud_rate, fs):
    #sample_len = 1/fs
    #alpha_len = 1/baud_rate
    #upsample_ratio = alpha_len/sample_len
    #alphabet_us = {}
    #for key, value in alphabet.items():
        #val_us = ""
        #for i in value:
            #for j in range(0, int(upsample_ratio)):
                #val_us = val_us + i
        #alphabet_us[key] = val_us
    #return alphabet_us, upsample_ratio

def wpm_to_symbol_len(wpm):
    # from https://www.eham.net/ehamforum/smf/index.php?topic=8534.0;wap2
    return 1200/wpm

def bin_string_to_numpy_array(bin_string):
    tmp = np.zeros(len(bin_string))
    for i in range(0, len(bin_string)):
        tmp[i] = int(bin_string[i])
    tmp[tmp == 0] = -1
    return tmp

def correlate_value(in1, in2):
    if len(in1) != len(in2):
        return 0
    #print(len(in1), len(in2))
    out = np.zeros(len(in1))
    for i in range(0, len(in1)):
        out[i] = in1[i] * in2[i]
    
    return np.sum(out)/len(out)

#def correlate_alphabet(bitstream, alphabet):
    #labels = []
    #values = []
    #for key, value in alphabet.items():
        #labels.append(key)
        #values.append(value)
    
    #correlator_outputs = np.zeros(len(labels))
    #for i in range(0, len(values)):
        #tmp_template = bin_string_to_numpy_array(values[i])
        #if len(bitstream) >= len(tmp_template):
            #tmp_data = bitstream[0:len(tmp_template)]
            
            #val = correlate_value(tmp_template, tmp_data)/len(tmp_template)
            #correlator_outputs[i] = val
        #else:
            #correlator_outputs[i] = 0
    
    #max_val = np.argmax(correlator_outputs)
    #return labels[max_val], len(values[max_val])
        
    

#def decode_block(bitstream, sync, alphabet_data):
    #alphabet = alphabet_data[0]
    #upsample_ratio = alphabet_data[1]
    #bitstream_temp = bitstream[sync[0]:sync[1]]
    #offset = 0
    #while True:
        #correlator_out = correlate_alphabet(bitstream_temp[offset:len(bitstream_temp)], alphabet)
        #offset = offset + correlator_out[1]
        
        #print("Val:%s, Offset:%i" % (correlator_out[0], offset))
        #if len(bitstream_temp)-offset < 0:
            #break
        
        #while True:
            #if bitstream_temp[offset] == 1:
                #break
            #else:
                #offset = offset + 1
                
#def decode_block(bitstream, sync, alphabet_data, debug = False):
    
    ### Try again - search for gaps? Yeah - use np.correlate
    #bitstream_temp = bitstream[sync[0]:sync[1]]
    
    #wpm = 22
    #ts = wpm_to_symbol_len(wpm)
    #fs = 1000
    #symbol_len = (fs * ts)/1000
    #alphabet = alphabet_data[0]
    
    #char_gap_array = bin_string_to_numpy_array(repeat_to_length("0", int(symbol_len*3)))
    
    #np.save("bitstream.npy", bitstream_temp)
    
    #plt.plot(bitstream_temp)
    #plt.show()

        
        
        
def decode_block(bitstream, alphabet, wpm):
    #bitstream = np.load("bitstream.npy")
    
    #wpm = 22
    #ts = wpm_to_symbol_len(wpm)
    ts = int(1200/wpm) # In ms
    fs = 1000
    symbol_len = (fs * ts)/1000
    
    offset = 0
    while True:
        correlator_output = correlate_alphabet(bitstream[0 + offset:22*ts + offset], alphabet, ts)
        if correlator_output == None:
            print("\n### Decode Complete ###")
            return
        print(correlator_output[0], end='', flush=True)
        offset = offset + correlator_output[1]
    
        
    return
        
def correlate_alphabet(bits, alphabet, ts):
    alphabet_keys = list(alphabet.keys())
    alphabet_values = list(alphabet.values())
    
    alphabet_values_us = []
    
    for i in range(0, len(alphabet_keys)):
        tmp_str = ""
        for j in range(0, len(alphabet_values[i])):
            tmp_str = tmp_str + repeat_to_length(alphabet_values[i][j], int(ts))
        alphabet_values_us.append(bin_string_to_numpy_array(tmp_str))
    
    
    ans = np.zeros(len(alphabet_values_us))
    for i in range(0, len(alphabet_values_us)):
        ans[i] = correlate_value(alphabet_values_us[i], bits[0:len(alphabet_values_us[i])])
    
    correlator_result = np.argmax(ans)
    
    #plt.plot(bits)
    #plt.plot(alphabet_values_us[18])
    #plt.show()
    
    for offset_delta in range(len(alphabet_values_us[correlator_result]) - 22, len(bits)):
        if bits[offset_delta] == 1:
            return alphabet_keys[correlator_result], offset_delta
    
    
    
        

#def correlate_debits(bitstream, symbol_len):
    #bits = [None, None, None, None]
    #bits[0] = bin_string_to_numpy_array(repeat_to_length("0", int(symbol_len)*2))
    #bits[1] = bin_string_to_numpy_array(repeat_to_length("0", int(symbol_len)) + repeat_to_length("1", int(symbol_len)))
    #bits[2] = bin_string_to_numpy_array(repeat_to_length("1", int(symbol_len)) + repeat_to_length("0", int(symbol_len)))
    #bits[3] = bin_string_to_numpy_array(repeat_to_length("1", int(symbol_len)*2))   
    
    #labels = ["00", "01", "10", "11"]
    
    #ans = np.zeros(len(bits))
    #for i in range(0, len(bits)):
        #ans[i] = correlate_value(bitstream, bits[i])
    
    #return labels[np.argmax(ans)]

# From -->> https://stackoverflow.com/questions/3391076/repeat-string-to-certain-length
def repeat_to_length(string_to_expand, length):
    return string_to_expand*length

def plot_mag_data(mag):
    
    for i in mag:
        plt.plot(i)
    plt.show()
    return
