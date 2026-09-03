import sys
import time
import numpy as np
import sounddevice as sd
from omnivad.stream_vad import OmniStreamVAD

print("Loading OmniVAD (PyTorch) Streaming Model...")
try:
    vad = OmniStreamVAD()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 160  # 10ms

def print_bar(prob, rms):
    W = 40
    p = max(0.0, min(1.0, prob))
    
    # Noise Gate
    if rms < 0.002:
        p = 0.0
        
    filled = int(p * W)
    bar = '#' * filled + '.' * (W - filled)
    
    if p >= 0.5:
        lbl, col = "SPEECH", "\033[92m"
    elif p >= 0.3:
        lbl, col = "maybe ", "\033[93m"
    else:
        lbl, col = "silent", "\033[91m"
    rst = "\033[0m"
    
    sys.stdout.write(f"\r  {col}[{bar}]{rst}  {p:.2f}  [{col}{lbl}{rst}]  Vol: {rms:.4f}   ")
    sys.stdout.flush()

audio_buffer = []

def audio_callback(indata, frames, time_info, status):
    if status:
        pass
    mono = np.mean(indata, axis=1)
    audio_buffer.extend(mono.tolist())

print("\n  ============================================")
print("  PyTorch VAD - LIVE MICROPHONE TEST (15 sec)")
print("  ============================================")
print("  Speak into your microphone now!\n")

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)

try:
    with stream:
        start_time = time.time()
        while time.time() - start_time < 15:
            if len(audio_buffer) >= CHUNK_SAMPLES:
                chunk = np.array(audio_buffer[:CHUNK_SAMPLES], dtype=np.float32)
                del audio_buffer[:CHUNK_SAMPLES]
                
                rms = np.sqrt(np.mean(chunk**2))
                
                result = vad.process(chunk)
                if result is not None:
                    print_bar(result.confidence, rms)
            else:
                time.sleep(0.001)
except Exception as e:
    print(e)
print("\nDone!")
