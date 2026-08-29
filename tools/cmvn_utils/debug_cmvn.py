#!/usr/bin/env python3
import struct
from pathlib import Path

data = open('models_pth/vad/cmvn.ark', 'rb').read()
print(f'Total: {len(data)} bytes\n')

offset = 0
print(f'Offset {offset:3d}: Header {repr(data[offset:offset+5])}')
offset += 5

print(f'Offset {offset:3d}: Format {data[offset:offset+5].hex()}')
offset += 5

print(f'Offset {offset:3d}: Marker 0x{data[offset]:02x}')
offset += 1

dim = struct.unpack('<I', data[offset:offset+4])[0]
print(f'Offset {offset:3d}: Dimension {dim}')
offset += 4

print(f'Offset {offset:3d}: Mean data ({dim} floats = {dim*4} bytes)')
mean_floats = struct.unpack(f'<{dim}f', data[offset:offset+dim*4])
print(f'  First 5 mean values: {mean_floats[:5]}')
print(f'  Last 5 mean values: {mean_floats[-5:]}')
offset += dim * 4

print(f'\nOffset {offset:3d}: Next marker 0x{data[offset]:02x}')
print(f'  (0x04 = dimension marker, 0xfd = ???)')

if data[offset] == 0xfd:
    # This might be "SPARSE" or "DOUBLE" marker in Kaldi
    # Skip and try next byte
    offset += 1
    print(f'Offset {offset:3d}: After 0xfd marker: 0x{data[offset]:02x}')
    
if data[offset] == 0x04:
    offset += 1
    var_dim = struct.unpack('<I', data[offset:offset+4])[0]
    print(f'Offset {offset:3d}: Variance dimension {var_dim}')
    offset += 4
    
    print(f'Offset {offset:3d}: Variance data ({var_dim} floats = {var_dim*4} bytes)')
    var_floats = struct.unpack(f'<{var_dim}f', data[offset:offset+var_dim*4])
    print(f'  First 5 variance values: {var_floats[:5]}')
    print(f'  Last 5 variance values: {var_floats[-5:]}')
