#!/usr/bin/env python3
import struct

data = open('models_pth/vad/cmvn.ark', 'rb').read()
print(f'Searching in {len(data)} bytes for 162 reasonable floats (81 mean + 81 variance)...\n')

found = False
for offset in range(0, len(data) - 648, 4):  # 162*4 = 648 bytes
    try:
        floats = struct.unpack('<162f', data[offset:offset+648])
        
        # Check if mean values look reasonable
        mean_ok = all(-50 < f < 50 for f in floats[:81])
        
        # Check if variance values look reasonable
        var_ok = all(0.01 < f < 1000 for f in floats[81:162])
        
        if mean_ok and var_ok:
            print(f'[OK] Found at offset {offset}:')
            print(f'  Mean range: [{min(floats[:81]):.3f}, {max(floats[:81]):.3f}]')
            print(f'  Var range: [{min(floats[81:162]):.3f}, {max(floats[81:162]):.3f}]')
            print(f'  First 5 mean: {[f"{v:.3f}" for v in floats[:5]]}')
            print(f'  First 5 var: {[f"{v:.3f}" for v in floats[81:86]]}')
            print(f'\n  Using offset {offset} for CMVN parsing')
            found = True
            break
    except:
        pass

if not found:
    print('[FAIL] No reasonable float ranges found')
