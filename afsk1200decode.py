#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#File        afsk.....py
#Author      Jonathan Rawlinson/M0ZJO
#Date        13/01/2019
#Desc.       Desc.....

# Import util libs
import sys, getopt

help_txt = 'pyrscw.py -i <inputfile> -w <wpm> -o <work_id>'

def main(argv):
    inputfile = ''
    wpm = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","wpm=","work_id="])
    except getopt.GetoptError:
        print("Options error:")
        print(help_txt)
        sys.exit(2)
        
    if len(args) == 0:
        print("Options error:")
        print(help_txt)
        
    for opt, arg in opts:
        if opt == '-h':
            print(help_txt)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--work_id"):
            work_id = arg
    print("### Input file: ", inputfile)
    print("### Work ID is:", work_id)  
    
    # Import the DSP library
    import afsk1200lib
    from afsk1200lib import log

    # Set up variables
    min_signal_len = 2000 #In ms
    threshold_value = 0.1 #Normalised to varience of signal
    internal_resample_val = None #Internal fs for initial DSP (Reduce for speed)
    detection_threshold = 0.1 #Value to determine if a signal is present

    afsk1200lib.print_header()

    # Open file and load into array
    log("### Opening file ###")
    wav_file = afsk1200lib.open_wav_file(inputfile, resample = internal_resample_val)

    # Remove any DC present in the array.
    # TODO this should be a windowed method to run over the file
    log("### Remove DC ###")
    wav_file = afsk1200lib.remove_dc(wav_file)
    
    bitstream = afsk1200lib.fsk_demodulate(wav_file.data, 1200, 1800, wav_file.rate, 1200)

    ascii_out = afsk1200lib.decode_block(bitstream)
    
    print(ascii_out)
    # Designs a FIR filter to remove non-CW components
    #log("### Applying filters ###")
    #filtered_data = afsk1200lib.generate_filtered_baseband(wav_file)
    
    #afsk1200lib.plot_numpy_data([filtered_data[0], filtered_data[1]])

    ## Dowsample full rate file to 1000Hz
    #log("### Downsampling ###")
    #downsampled_power_data = afsk1200lib.downsample_abs_bb(baseband_data, wav_file.rate)

    # Calculate the magnitude vector of the IQ
    #log("### Calculate Mag ###")
    #magnitude_data_low = afsk1200lib.compute_abs(filtered_data[0])
    #magnitude_data_high = afsk1200lib.compute_abs(filtered_data[1])
    
    #log("### Filter Discriminator Output ###")
    #filt_d = afsk1200lib.filter_discriminator(magnitude_data_low, magnitude_data_high, fs = wav_file.rate)
    
    #log("### Threshold Bits ###")
    #bits_os = afsk1200lib.quantise(filt_d)
    
    ##afsk1200lib.plot_numpy_data([bits_os])
    
    #idx, ctr = afsk1200lib.PLL(bits_os)
    
    #bits = bits_os[idx]
    
    #afsk1200lib.plot_numpy_data([magnitude_data_low[0:4800], magnitude_data_high[0:4800]])
    
    #extractdata_block = afsk1200lib.decode_block(bits[1000:len(bits)])

if __name__ == "__main__":
   main(sys.argv[1:])

