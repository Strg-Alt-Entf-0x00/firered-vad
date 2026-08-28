#!/usr/bin/env python3
"""
PyTorch Inference for FireRed-VAD Models
Loads original PTH models and runs inference for golden test generation
"""

import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models_converter.pytorch_loader import PyTorchModelLoader


class FireRedVADPyTorch:
    """PyTorch inference wrapper for FireRed-VAD models"""
    
    def __init__(self, model_path: Path, cmvn_path: Path, mode: str = "vad"):
        """
        Initialize PyTorch model
        
        Args:
            model_path: Path to model.pth.tar
            cmvn_path: Path to cmvn.ark
            mode: Model mode (vad, stream-vad, aed)
        """
        self.model_path = model_path
        self.cmvn_path = cmvn_path
        self.mode = mode
        
        # Load model
        loader = PyTorchModelLoader(model_path, cmvn_path)
        self.weights = loader.load()
        self.mean, self.variance = loader.load_cmvn()
        
        print(f"Loaded PyTorch model: {mode}")
        print(f"  Tensors: {len(self.weights)}")
        print(f"  Parameters: {sum(w.size for w in self.weights.values()):,}")
    
    def extract_features(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract acoustic features (Fbank) from audio
        
        Args:
            audio: Audio samples (1D array)
            sample_rate: Sample rate (must be 16kHz)
            
        Returns:
            Features array (n_frames, 80)
        """
        if sample_rate != 16000:
            raise ValueError("Sample rate must be 16kHz")
        
        # Convert to torch tensor
        waveform = torch.from_numpy(audio).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        
        # Extract Fbank features (80-dim)
        # Frame length = 25ms, Frame shift = 10ms
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=sample_rate,
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            dither=0.0,
            window_type='hamming'
        )
        
        return fbank.numpy()
    
    def apply_cmvn(self, features: np.ndarray) -> np.ndarray:
        """
        Apply CMVN (Cepstral Mean and Variance Normalization)
        
        Args:
            features: Input features (n_frames, 80)
            
        Returns:
            Normalized features (n_frames, 80)
        """
        # CMVN: (x - mean) / sqrt(variance)
        normalized = (features - self.mean) / np.sqrt(self.variance + 1e-8)
        return normalized.astype(np.float32)
    
    def forward_pass(self, features: np.ndarray) -> np.ndarray:
        """
        Run forward pass through DFSMN network
        
        NOTE: This is a simplified placeholder. Real implementation requires:
        - Proper DFSMN architecture (memory blocks, convolutions)
        - Layer-by-layer forward pass
        - Activation functions (ReLU, Sigmoid)
        
        Args:
            features: Input features (n_frames, 80)
            
        Returns:
            Output probabilities (n_frames, n_classes)
        """
        raise NotImplementedError(
            "Full PyTorch DFSMN forward pass requires model architecture definition. "
            "This is complex and should be implemented based on FireRedTeam's code. "
            "For now, use this as a placeholder for golden test infrastructure."
        )
    
    def predict(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Run full inference pipeline
        
        Args:
            audio: Audio samples
            sample_rate: Sample rate
            
        Returns:
            Predictions (n_frames, n_classes)
        """
        # Extract features
        features = self.extract_features(audio, sample_rate)
        
        # Apply CMVN
        features = self.apply_cmvn(features)
        
        # Forward pass
        predictions = self.forward_pass(features)
        
        return predictions
    
    def predict_file(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Run inference on audio file
        
        Args:
            audio_path: Path to WAV file
            
        Returns:
            Tuple of (predictions, sample_rate)
        """
        # Load audio with scipy (no FFmpeg needed)
        try:
            from scipy.io import wavfile
            sample_rate, audio = wavfile.read(str(audio_path))
            
            # Convert to float32 [-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            
            # Take first channel if stereo
            if audio.ndim > 1:
                audio = audio[:, 0]
                
        except ImportError:
            # Fallback to wave module (standard library)
            import wave
            with wave.open(str(audio_path), 'rb') as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)
                
                # Convert to numpy array
                if wf.getsampwidth() == 2:  # 16-bit
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    raise ValueError(f"Unsupported sample width: {wf.getsampwidth()}")
                
                # Take first channel if stereo
                if wf.getnchannels() > 1:
                    audio = audio[::wf.getnchannels()]
        
        # Run inference
        predictions = self.predict(audio, sample_rate)
        
        return predictions, sample_rate


def load_model(model_type: str, pth_dir: Path) -> FireRedVADPyTorch:
    """
    Load PyTorch model by type
    
    Args:
        model_type: vad, stream-vad, or aed
        pth_dir: Base directory containing PTH models
        
    Returns:
        FireRedVADPyTorch instance
    """
    model_types = {
        "vad": "vad",
        "stream-vad": "stream-vad",
        "aed": "aed"
    }
    
    if model_type not in model_types:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_dir = pth_dir / model_types[model_type]
    model_path = model_dir / "model.pth.tar"
    cmvn_path = model_dir / "cmvn.ark"
    
    return FireRedVADPyTorch(model_path, cmvn_path, model_type)


def test_feature_extraction():
    """Test feature extraction on example audio"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference_pytorch.py <audio.wav>")
        sys.exit(1)
    
    audio_path = Path(sys.argv[1])
    pth_dir = Path("../../pth_models").resolve()
    
    # Fix path - pth_models is in firered-vad root
    if not pth_dir.exists():
        pth_dir = Path(__file__).parent.parent.parent / "pth_models"
    
    print(f"Testing feature extraction on: {audio_path}")
    print()
    
    try:
        # Load model
        model = load_model("vad", pth_dir)
        
        # Load audio with scipy or wave (no FFmpeg needed)
        try:
            from scipy.io import wavfile
            sample_rate, audio = wavfile.read(str(audio_path))
            
            # Convert to float32 [-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            
            # Take first channel if stereo
            if audio.ndim > 1:
                audio = audio[:, 0]
                
        except ImportError:
            # Fallback to wave module
            import wave
            with wave.open(str(audio_path), 'rb') as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)
                
                if wf.getsampwidth() == 2:  # 16-bit
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    raise ValueError(f"Unsupported sample width: {wf.getsampwidth()}")
                
                # Take first channel if stereo
                if wf.getnchannels() > 1:
                    audio = audio[::wf.getnchannels()]
        
        print(f"Audio loaded:")
        print(f"  Duration: {len(audio) / sample_rate:.2f}s")
        print(f"  Samples: {len(audio):,}")
        print(f"  Sample rate: {sample_rate} Hz")
        print()
        
        # Extract features
        features = model.extract_features(audio, sample_rate)
        
        print(f"Features extracted:")
        print(f"  Shape: {features.shape}")
        print(f"  Frames: {features.shape[0]}")
        print(f"  Dimensions: {features.shape[1]}")
        print(f"  Range: [{features.min():.3f}, {features.max():.3f}]")
        print()
        
        # Apply CMVN
        normalized = model.apply_cmvn(features)
        
        print(f"CMVN applied:")
        print(f"  Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        print(f"  Mean: {normalized.mean():.6f}")
        print(f"  Std: {normalized.std():.6f}")
        print()
        
        print("✓ Feature extraction working!")
        print()
        print("NOTE: Full inference requires DFSMN architecture implementation.")
        print("This would need to be ported from FireRedTeam's original code.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_feature_extraction()
