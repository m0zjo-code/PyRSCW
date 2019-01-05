import pyrscwlib

wpm = 22
min_signal_len = 2000 #In ms

print("### Decode ###")
output_list = pyrscwlib.decode_block_debug(pyrscwlib.generate_alphabet())
