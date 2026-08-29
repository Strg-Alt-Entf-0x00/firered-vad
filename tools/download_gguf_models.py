#!/usr/bin/env python3
"""
GGUF Model Downloader for FireRed-VAD
Downloads pre-converted GGUF models from HuggingFace

Downloads from nested HuggingFace structure and saves flat in local directory.
"""

import argparse
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download


# HuggingFace repository
REPO_ID = "Strg-Alt-Entf-0x00/FireRedVAD-GGUF"

# Model configurations
MODEL_TYPES = ['vad', 'stream-vad', 'aed']
QUANT_TYPES = ['fp32', 'int16', 'int8', 'int8-ch']

# File patterns
FILE_PATTERNS = {
    'vad': 'firered-vad',
    'stream-vad': 'firered-stream-vad',
    'aed': 'firered-aed'
}


def download_file(repo_id: str, remote_path: str, local_dir: Path, filename: str):
    """
    Download a single file from HuggingFace
    
    Args:
        repo_id: HuggingFace repo ID
        remote_path: Path in repo (e.g., "vad/fp32/file.gguf")
        local_dir: Local output directory
        filename: Local filename to save as
    """
    try:
        print(f"  [DOWNLOAD] {remote_path}...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        # Move from nested structure to flat structure
        downloaded = Path(downloaded_path)
        target = local_dir / filename
        
        if downloaded != target:
            # Create parent directory
            target.parent.mkdir(parents=True, exist_ok=True)
            # Move file
            downloaded.replace(target)
            
            # Clean up empty directories
            try:
                downloaded.parent.rmdir()
                downloaded.parent.parent.rmdir()
                downloaded.parent.parent.parent.rmdir()
            except:
                pass  # Ignore if not empty
        
        print(f"  [OK] {filename}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to download {remote_path}: {e}")
        return False


def download_models(
    model_types: list,
    quant_types: list,
    output_dir: Path,
    include_debug: bool = True
):
    """
    Download GGUF models from HuggingFace
    
    Args:
        model_types: List of model types to download
        quant_types: List of quantization types to download
        output_dir: Output directory (flat structure)
        include_debug: Whether to download debug.json files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Models to download: {', '.join(model_types)}")
    print(f"Quantizations: {', '.join(quant_types)}")
    print()
    
    downloaded = 0
    failed = 0
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Downloading {model_type.upper()} models...")
        print(f"{'='*70}")
        
        file_prefix = FILE_PATTERNS[model_type]
        
        for quant in quant_types:
            # GGUF file
            gguf_filename = f"{file_prefix}-{quant}.gguf"
            remote_gguf = f"{model_type}/{quant}/{gguf_filename}"
            
            if download_file(REPO_ID, remote_gguf, output_dir, gguf_filename):
                downloaded += 1
            else:
                failed += 1
            
            # Debug JSON file (optional)
            if include_debug:
                json_filename = f"{file_prefix}-{quant}-debug.json"
                remote_json = f"{model_type}/{quant}/{json_filename}"
                
                if download_file(REPO_ID, remote_json, output_dir, json_filename):
                    downloaded += 1
                else:
                    failed += 1
    
    return downloaded, failed


def list_available_models():
    """List all available models"""
    print("="*70)
    print("Available FireRed-VAD GGUF Models")
    print("="*70)
    print()
    print(f"Repository: https://huggingface.co/{REPO_ID}")
    print()
    
    for model_type in MODEL_TYPES:
        file_prefix = FILE_PATTERNS[model_type]
        print(f"\n[{model_type}]")
        print(f"  Prefix: {file_prefix}")
        print(f"  Quantizations:")
        for quant in QUANT_TYPES:
            size = {
                'fp32': '~2.3 MB',
                'int16': '~1.2 MB',
                'int8': '~600 KB',
                'int8-ch': '~600 KB'
            }.get(quant, 'Unknown')
            print(f"    - {quant:8s} ({size})")
    
    print()
    print(f"Total: {len(MODEL_TYPES)} model types × {len(QUANT_TYPES)} quantizations = {len(MODEL_TYPES) * len(QUANT_TYPES)} models")
    print("Plus debug.json files = 24 files total")
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Download FireRed-VAD GGUF models from HuggingFace",
        epilog="""
Examples:
  python download_gguf_models.py                           # Download all models
  python download_gguf_models.py --model vad               # Download only VAD models
  python download_gguf_models.py --quant int8-ch           # Download only INT8-CH models
  python download_gguf_models.py --model stream-vad --quant fp32  # Specific model
  python download_gguf_models.py --list                    # List available models
  python download_gguf_models.py --no-debug                # Skip debug.json files
        """
    )
    parser.add_argument(
        '--model',
        choices=['vad', 'stream-vad', 'aed', 'all'],
        default='all',
        help='Model type to download (default: all)'
    )
    parser.add_argument(
        '--quant',
        choices=['fp32', 'int16', 'int8', 'int8-ch', 'all'],
        default='all',
        help='Quantization type to download (default: all)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent.parent / 'models_gguf',
        help='Output directory for GGUF models (default: ../models_gguf)'
    )
    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Skip downloading debug.json files'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models and exit'
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list:
        list_available_models()
        return 0
    
    print("="*70)
    print("FireRed-VAD GGUF Model Downloader")
    print("="*70)
    print(f"Repository: {REPO_ID}")
    print()
    
    # Determine what to download
    model_types = MODEL_TYPES if args.model == 'all' else [args.model]
    quant_types = QUANT_TYPES if args.quant == 'all' else [args.quant]
    
    # Download
    try:
        downloaded, failed = download_models(
            model_types=model_types,
            quant_types=quant_types,
            output_dir=args.output,
            include_debug=not args.no_debug
        )
        
        print()
        print("="*70)
        print("Download Summary")
        print("="*70)
        print(f"Successfully downloaded: {downloaded} files")
        print(f"Failed: {failed} files")
        print(f"Output directory: {args.output}")
        print()
        
        if failed > 0:
            print("[WARN] Some files failed to download")
            return 1
        else:
            print("[SUCCESS] All files downloaded successfully!")
            return 0
            
    except Exception as e:
        print()
        print(f"[ERROR] Download failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
