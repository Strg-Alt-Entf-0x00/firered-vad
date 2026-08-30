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
        waveform = torch.from_numpy(audio).float() * 32768.0
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
        
        DFSMN Architecture (Deep Feed-Forward Sequential Memory Network):
        1. Input: (n_frames, 80) Fbank features (CMVN normalized)
        2. FC1 layer: (80 → hidden_dim) with ReLU
        3. FC2 layer: (hidden_dim → hidden_dim) with ReLU
        4. FSMN1: Lookback/lookahead filters (memory mechanism)
        5. FSMN blocks 0-6: Each has:
           - FC1: (hidden_dim → hidden_dim) with ReLU
           - FC2: (hidden_dim → hidden_dim) linear
           - FSMN: Lookback/lookahead filters
           - Residual connection
        6. DNN layer: (hidden_dim → hidden_dim) with ReLU
        7. Output layer: (hidden_dim → n_classes) with Sigmoid
        
        Args:
            features: Input features (n_frames, 80)
            
        Returns:
            Output probabilities (n_frames, n_classes)
        """
        # Convert to torch tensor
        x = torch.from_numpy(features).float()
        n_frames = x.shape[0]
        
        # Layer 1: FC1 (80 → 256)
        fc1_weight = torch.from_numpy(self.weights['dfsmn.fc1.0.weight']).float()
        fc1_bias = torch.from_numpy(self.weights['dfsmn.fc1.0.bias']).float()
        x = torch.matmul(x, fc1_weight.T) + fc1_bias
        x = torch.relu(x)
        # DEBUG: print(f"After FC1: {x.shape}")
        
        # Layer 2: FC2 (256 → 128)
        fc2_weight = torch.from_numpy(self.weights['dfsmn.fc2.0.weight']).float()
        fc2_bias = torch.from_numpy(self.weights['dfsmn.fc2.0.bias']).float()
        x = torch.matmul(x, fc2_weight.T) + fc2_bias
        x = torch.relu(x)
        # DEBUG: print(f"After FC2: {x.shape}")
        
        # Layer 3: FSMN1 (lookback/lookahead filters) - operates on 128-dim
        x = self._apply_fsmn_layer(
            x,
            'dfsmn.fsmn1.lookback_filter.weight',
            'dfsmn.fsmn1.lookahead_filter.weight'
        )
        
        # Layers 4-10: FSMN blocks 0-6 (each operates on 128-dim input/output)
        for i in range(7):
            x = self._apply_fsmn_block(x, i)
        
        # Layer 11: DNN layer
        dnn_weight = torch.from_numpy(self.weights['dfsmn.dnns.0.weight']).float()
        dnn_bias = torch.from_numpy(self.weights['dfsmn.dnns.0.bias']).float()
        x = torch.matmul(x, dnn_weight.T) + dnn_bias
        x = torch.relu(x)
        print(f"[PyTorch DEBUG] After DNN: shape={x.shape}, mean={x.mean():.6f}, first_frame_mean={x[0].mean():.6f}")
        
        # Layer 12: Output layer
        out_weight = torch.from_numpy(self.weights['out.weight']).float()
        out_bias = torch.from_numpy(self.weights['out.bias']).float()
        x = torch.matmul(x, out_weight.T) + out_bias
        print(f"[PyTorch DEBUG] Pre-sigmoid: shape={x.shape}, mean={x.mean():.6f}, first_frame={x[0,0]:.6f}")
        x = torch.sigmoid(x)
        print(f"[PyTorch DEBUG] Post-sigmoid: shape={x.shape}, mean={x.mean():.6f}, first_frame={x[0,0]:.6f}")
        
        return x.numpy()
    
    def _apply_fsmn_layer(
        self,
        x: torch.Tensor,
        lookback_key: str,
        lookahead_key: str
    ) -> torch.Tensor:
        """
        Apply FSMN (Feed-Forward Sequential Memory Network) layer
        
        FSMN uses 1D convolutions with lookback and lookahead filters
        to model temporal dependencies without recurrence.
        
        Args:
            x: Input tensor (n_frames, hidden_dim)
            lookback_key: Lookback filter weight key
            lookahead_key: Lookahead filter weight key (may not exist for streaming)
            
        Returns:
            Output tensor (n_frames, hidden_dim)
        """
        n_frames, input_dim = x.shape
        
        # Load filters - shape is (hidden_dim, 1, filter_order)
        lookback_filter = torch.from_numpy(self.weights[lookback_key]).float()
        
        # Extract dimensions from filter shape
        filter_hidden_dim = lookback_filter.shape[0]
        lookback_order = lookback_filter.shape[-1]
        
        # Reshape filter to (hidden_dim, 1, filter_order) if needed
        if lookback_filter.dim() == 3:
            lookback_filter = lookback_filter.squeeze(1)  # (hidden_dim, filter_order)
        
        # Reshape input: (n_frames, hidden_dim) → (1, hidden_dim, n_frames)
        x_conv = x.T.unsqueeze(0)  # (1, input_dim, n_frames)
        
        # Pad for lookback (pad left side - past context)
        x_padded = torch.nn.functional.pad(x_conv, (lookback_order - 1, 0), mode='constant', value=0)
        
        # Apply lookback convolution (grouped conv1d)
        lookback_out = torch.nn.functional.conv1d(
            x_padded,
            lookback_filter.unsqueeze(1),  # (hidden_dim, 1, filter_order)
            groups=filter_hidden_dim
        ).squeeze(0).T  # (n_frames, hidden_dim)
        
        # Lookahead filter (if exists - not for streaming models)
        if lookahead_key in self.weights:
            lookahead_filter = torch.from_numpy(self.weights[lookahead_key]).float()
            lookahead_order = lookahead_filter.shape[-1]
            
            # Reshape filter
            if lookahead_filter.dim() == 3:
                lookahead_filter = lookahead_filter.squeeze(1)
            
            # Pad for lookahead (pad right side - future context)
            x_padded_ahead = torch.nn.functional.pad(
                x_conv,
                (0, lookahead_order - 1),
                mode='constant',
                value=0
            )
            
            # Apply lookahead convolution (reverse filter)
            lookahead_out = torch.nn.functional.conv1d(
                x_padded_ahead,
                lookahead_filter.flip(dims=[-1]).unsqueeze(1),
                groups=filter_hidden_dim
            ).squeeze(0).T  # (n_frames, hidden_dim)
            
            # Combine lookback + lookahead + input (residual)
            output = x + lookback_out + lookahead_out
        else:
            # Streaming mode: only lookback + input (residual)
            output = x + lookback_out
        
        return output
    
    def _apply_fsmn_block(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        """
        Apply one FSMN block
        
        Architecture per block:
        - FC1: (128 → 256) with ReLU (expansion)
        - FC2: (256 → 128) linear (contraction)
        - FSMN: Lookback/lookahead filters on 128-dim
        - Residual: Add input to output
        
        Args:
            x: Input tensor (n_frames, 128)
            block_idx: Block index (0-6)
            
        Returns:
            Output tensor (n_frames, 128)
        """
        residual = x
        
        # FC1: (128 → 256) with ReLU
        fc1_weight = torch.from_numpy(self.weights[f'dfsmn.fsmns.{block_idx}.fc1.0.weight']).float()
        fc1_bias = torch.from_numpy(self.weights[f'dfsmn.fsmns.{block_idx}.fc1.0.bias']).float()
        x = torch.matmul(x, fc1_weight.T) + fc1_bias
        x = torch.relu(x)
        
        # FC2: (256 → 128) linear
        fc2_weight = torch.from_numpy(self.weights[f'dfsmn.fsmns.{block_idx}.fc2.weight']).float()
        x = torch.matmul(x, fc2_weight.T)
        
        # FSMN: Lookback/lookahead filters on 128-dim
        lookback_key = f'dfsmn.fsmns.{block_idx}.fsmn.lookback_filter.weight'
        lookahead_key = f'dfsmn.fsmns.{block_idx}.fsmn.lookahead_filter.weight'
        x = self._apply_fsmn_layer(x, lookback_key, lookahead_key)
        
        # Residual connection
        x = x + residual
        
        return x
    
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
    pth_dir = Path("../../models_pth").resolve()
    
    # Fix path - models_pth is in firered-vad root
    if not pth_dir.exists():
        pth_dir = Path(__file__).parent.parent.parent / "models_pth"
    
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
        
        print("[OK] Feature extraction working!")
        print()
        
        # Test full inference
        print("Testing full DFSMN inference...")
        try:
            predictions = model.predict(audio, sample_rate)
            
            print(f"Inference complete:")
            print(f"  Output shape: {predictions.shape}")
            print(f"  Predictions: {predictions.shape[0]} frames x {predictions.shape[1]} classes")
            print(f"  Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
            print(f"  Mean: {predictions.mean():.6f}")
            print()
            
            # Show first/last few predictions
            print("Sample predictions (first 10 frames):")
            for i in range(min(10, predictions.shape[0])):
                print(f"  Frame {i:4d}: {predictions[i]}")
            
            print()
            print("[OK] Full DFSMN inference working!")
            
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_feature_extraction()
