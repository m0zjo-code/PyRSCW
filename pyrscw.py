import pyrscwlib

wpm = 22
min_signal_len = 2000 #In ms
print("### Opening file ###")
wav_file = pyrscwlib.open_wav_file("gqrx_20181230_160324_145727285.wav")

#pyrscwlib.plot_file(wav_file)

print("### Remove DC ###")
wav_file = pyrscwlib.remove_dc(wav_file)

print("### Find Carrier ###")
carrier_freq = pyrscwlib.find_carrier(wav_file)

print("### Generate Carriers ###")
carriers = pyrscwlib.generate_carriers(wav_file, carrier_freq)

print("### Applying carriers and filter ###")
baseband_data = pyrscwlib.generate_filtered_baseband(wav_file, carriers)

print("### Downsampling ###")
downsampled_power_data = pyrscwlib.downsample_abs_bb(baseband_data, wav_file.rate)

print("### Calculate Mag ###")
# Sample rate is now 1000Hz
magnitude_data = pyrscwlib.compute_abs(downsampled_power_data)

print("### Smooth ###")
magnitude_data_smoothed = pyrscwlib.smooth_mag(magnitude_data)

print("### Generate Threshold Vector ###")
thresh_vector = pyrscwlib.get_threshold(magnitude_data_smoothed)

print("### Apply Threshold ###")
shifted_mag = pyrscwlib.apply_threshold(magnitude_data_smoothed, thresh_vector)

print("### Detect Signal ###")
signal_present = pyrscwlib.signal_detect(magnitude_data)

print("### Quantise ###")
bitstream = pyrscwlib.quantise(shifted_mag)

print("### Bit Synch ###")
signal_synch_list = pyrscwlib.bit_synch(bitstream, signal_present, min_length = min_signal_len)

for i in signal_synch_list:
    print(i)

print("### Decode ###")
for i in range(0, len(signal_synch_list)):
    output_list = pyrscwlib.decode_block(bitstream[signal_synch_list[i][0]: signal_synch_list[i][1]], pyrscwlib.generate_alphabet(), wpm)

pyrscwlib.plot_mag_data([bitstream, signal_present])


