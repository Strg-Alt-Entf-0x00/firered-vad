#!/usr/bin/env python3
"""Analyze CMVN.ark format - try all possible Kaldi formats"""
import struct
from pathlib import Path

cmvn_path = Path("../pth_models/vad/cmvn.ark")
content = cmvn_path.read_bytes()

print(f"File size: {len(content)} bytes")
print(f"First 30 bytes (hex): {content[:30].hex()}")

# Since file is 1311 bytes and we need 2x80 doubles (1280 bytes)
# Plus ~31 bytes header = 1311, this matches!

# Kaldi stores CMVN as stats matrix, not mean/var directly
# The format is actually statistics that need to be converted

# Let's try a simpler approach - search for the actual data
print("\n" + "="*70)
print("Searching for valid mean/variance data...")
print("="*70)

# The data should be 160 doubles (1280 bytes)
# Try every possible offset
best_offset = None
for offset in range(0, len(content) - 1280):
    try:
        doubles = struct.unpack(f'<160d', content[offset:offset+1280])
        
        # Split into potential mean and variance
        mean = doubles[:80]
        var = doubles[80:160]
        
        # Check if this looks valid
        # Mean: should be reasonable audio feature values (-50 to 100)
        # Variance: should be positive (0.1 to 100)
        mean_valid = all(-100 < m < 150 for m in mean)
        var_valid = all(0.01 < v < 200 for v in var)
        
        if mean_valid and var_valid:
            print(f"\n✓ FOUND VALID DATA at offset {offset}:")
            print(f"  Mean[0:5]: {[f'{m:.3f}' for m in mean[:5]]}")
            print(f"  Mean[75:80]: {[f'{m:.3f}' for m in mean[75:80]]}")
            print(f"  Var[0:5]: {[f'{v:.3f}' for v in var[:5]]}")
            print(f"  Var[75:80]: {[f'{v:.3f}' for v in var[75:80]]}")
            print(f"  Mean range: [{min(mean):.3f}, {max(mean):.3f}]")
            print(f"  Var range: [{min(var):.3f}, {max(var):.3f}]")
            best_offset = offset
            break
    except:
        pass

if best_offset is not None:
    print(f"\n✓ Best offset for CMVN data: {best_offset}")
    print(f"  Header size: {best_offset} bytes")
else:
    print("\n✗ No valid CMVN data found")
    print("\nTrying as CMVN stats (sum, sum_sq, count) format...")
    
    # Alternative: Kaldi stores accumulation stats, not mean/var directly
    # Format: [count, sum[80], sum_sq[80]] or similar
    # We need to convert: mean = sum/count, var = sum_sq/count - mean^2
    
    for offset in range(0, min(100, len(content) - 650)):
        try:
            # Try reading count + vectors
            count = struct.unpack('<d', content[offset:offset+8])[0]
            if 100 < count < 1000000:  # Reasonable frame count
                sum_vec = struct.unpack('<80d', content[offset+8:offset+8+640])
                sum_sq_vec = struct.unpack('<80d', content[offset+8+640:offset+8+1280])
                
                # Convert to mean and variance
                mean = [s / count for s in sum_vec]
                var = [(sq / count - m*m) for sq, m in zip(sum_sq_vec, mean)]
                
                # Check validity
                mean_valid = all(-100 < m < 150 for m in mean[:10])
                var_valid = all(0.01 < v < 200 for v in var[:10]) and all(v > 0 for v in var)
                
                if mean_valid and var_valid:
                    print(f"\n✓ FOUND VALID STATS at offset {offset}:")
                    print(f"  Count: {count}")
                    print(f"  Mean[0:5]: {[f'{m:.3f}' for m in mean[:5]]}")
                    print(f"  Var[0:5]: {[f'{v:.3f}' for v in var[:5]]}")
                    print(f"  Mean range: [{min(mean):.3f}, {max(mean):.3f}]")
                    print(f"  Var range: [{min(var):.3f}, {max(var):.3f}]")
                    best_offset = offset
                    break
        except:
            pass

