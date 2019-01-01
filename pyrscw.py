import pyrscwlib

wpm = 22

wav_file = pyrscwlib.open_wav_file("gqrx_20181230_160324_145727285.wav")

#pyrscwlib.plot_file(wav_file)

wav_file = pyrscwlib.remove_dc(wav_file)

carrier_freq = pyrscwlib.find_carrier(wav_file)

carriers = pyrscwlib.generate_carriers(wav_file, carrier_freq)

baseband_data = pyrscwlib.generate_filtered_baseband(wav_file, carriers)

downsampled_power_data = pyrscwlib.downsample_abs_bb(baseband_data, wav_file.rate)

# Sample rate is now 1000Hz
magnitude_data = pyrscwlib.compute_abs(downsampled_power_data)

magnitude_data_smoothed = pyrscwlib.smooth_mag(magnitude_data)

thresh_vector = pyrscwlib.get_threshold(magnitude_data_smoothed)

shifted_mag = pyrscwlib.apply_threshold(magnitude_data_smoothed, thresh_vector)

signal_present = pyrscwlib.signal_detect(magnitude_data)

bitstream = pyrscwlib.quantise(shifted_mag)

signal_synch_list = pyrscwlib.bit_synch(bitstream, signal_present)

for i in signal_synch_list:
    print(i)

pyrscwlib.plot_mag_data([bitstream, signal_present])


