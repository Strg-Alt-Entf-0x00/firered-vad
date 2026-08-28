#!/usr/bin/env python3
"""
FireRed-VAD PTH Model Downloader
Downloads original PyTorch .pth.tar models from HuggingFace repository.

Usage:
    python download_pth_models.py                    # Download all models
    python download_pth_models.py --model vad        # Download specific model
    python download_pth_models.py --list             # List available models
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import hashlib


# Configuration
HUGGINGFACE_REPO = "FireRedTeam/FireRedVAD"
BASE_URL = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main"
# Default output directory relative to project root (one level up from tools/)
DEFAULT_OUTPUT_DIR = "../pth_models"

# Available models with metadata
MODELS = {
    "vad": {
        "name": "VAD (Non-Streaming)",
        "model_file": "model.pth.tar",
        "cmvn_file": "cmvn.ark",
        "size_mb": 2.37,
        "description": "Standard VAD - Highest accuracy (97.57% F1), offline processing",
        "url_prefix": "VAD"
    },
    "stream-vad": {
        "name": "Stream-VAD (Streaming)",
        "model_file": "model.pth.tar",
        "cmvn_file": "cmvn.ark",
        "size_mb": 2.26,
        "description": "Streaming VAD - Real-time, low latency (~96% F1)",
        "url_prefix": "Stream-VAD"
    },
    "aed": {
        "name": "AED (Audio Event Detection)",
        "model_file": "model.pth.tar",
        "cmvn_file": "cmvn.ark",
        "size_mb": 2.37,
        "description": "Audio Event Detection (speech/music/singing classification)",
        "url_prefix": "AED"
    }
}


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def disable():
        """Disable colors for Windows or non-TTY output"""
        Colors.HEADER = ''
        Colors.OKBLUE = ''
        Colors.OKCYAN = ''
        Colors.OKGREEN = ''
        Colors.WARNING = ''
        Colors.FAIL = ''
        Colors.ENDC = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''


def print_header(text: str) -> None:
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")


def print_success(text: str) -> None:
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}", file=sys.stderr)


def print_warning(text: str) -> None:
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text: str) -> None:
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """
    Display download progress bar
    
    Args:
        block_num: Current block number
        block_size: Size of each block in bytes
        total_size: Total file size in bytes
    """
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100.0 / total_size, 100.0)
        bar_length = 50
        filled_length = int(bar_length * downloaded // total_size)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        
        print(f'\r  Progress: |{bar}| {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)', end='')
        sys.stdout.flush()
        
        if downloaded >= total_size:
            print()  # New line after completion


def calculate_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Calculate SHA256 hash of a file
    
    Args:
        filepath: Path to file
        chunk_size: Size of chunks to read
        
    Returns:
        Hexadecimal SHA256 hash string
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file(url: str, output_path: Path, description: str, force: bool = False) -> bool:
    """
    Download a single file from URL
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        description: Description of the file being downloaded
        force: Force re-download even if file exists
        
    Returns:
        True if download successful, False otherwise
    """
    # Check if file already exists
    if output_path.exists() and not force:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print_warning(f"{description} already exists ({file_size_mb:.2f} MB)")
        return True
    
    print(f"\n{Colors.BOLD}Downloading:{Colors.ENDC} {description}")
    print(f"{Colors.BOLD}URL:{Colors.ENDC} {url}")
    
    try:
        # Download with progress bar
        urlretrieve(url, output_path, reporthook=progress_hook)
        
        # Verify file was downloaded
        if not output_path.exists():
            print_error("Download failed: File not created")
            return False
        
        # Check file size
        actual_size_mb = output_path.stat().st_size / (1024 * 1024)
        print_success(f"Downloaded: {output_path.name} ({actual_size_mb:.2f} MB)")
        
        return True
        
    except HTTPError as e:
        print_error(f"HTTP Error {e.code}: {e.reason}")
        if e.code == 404:
            print_error(f"File not found at: {url}")
        return False
        
    except URLError as e:
        print_error(f"Network Error: {e.reason}")
        print_info("Check your internet connection and try again")
        return False
        
    except KeyboardInterrupt:
        print_warning("\nDownload interrupted by user")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
            print_info("Cleaned up partial download")
        return False
        
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        return False


def download_model(model_key: str, output_dir: Path, force: bool = False) -> bool:
    """
    Download a model and its associated files
    
    Args:
        model_key: Model identifier (e.g., 'vad', 'stream-vad')
        output_dir: Directory to save the model
        force: Force re-download even if files exist
        
    Returns:
        True if download successful, False otherwise
    """
    if model_key not in MODELS:
        print_error(f"Unknown model: {model_key}")
        return False
    
    model_info = MODELS[model_key]
    model_dir = output_dir / model_key
    
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}Model:{Colors.ENDC} {model_info['name']}")
    print(f"{Colors.BOLD}Type:{Colors.ENDC} {model_key}")
    print(f"{Colors.BOLD}Size:{Colors.ENDC} ~{model_info['size_mb']} MB")
    print(f"{Colors.BOLD}Description:{Colors.ENDC} {model_info['description']}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    
    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download model file
    model_url = f"{BASE_URL}/{model_info['url_prefix']}/{model_info['model_file']}"
    model_path = model_dir / model_info['model_file']
    
    if not download_file(model_url, model_path, f"Model file ({model_info['model_file']})", force):
        return False
    
    # Download CMVN file
    cmvn_url = f"{BASE_URL}/{model_info['url_prefix']}/{model_info['cmvn_file']}"
    cmvn_path = model_dir / model_info['cmvn_file']
    
    if not download_file(cmvn_url, cmvn_path, f"CMVN file ({model_info['cmvn_file']})", force):
        return False
    
    # Calculate checksums
    print_info("Calculating checksums...")
    model_sha256 = calculate_sha256(model_path)
    cmvn_sha256 = calculate_sha256(cmvn_path)
    
    print(f"  Model SHA256: {model_sha256}")
    print(f"  CMVN SHA256:  {cmvn_sha256}")
    
    print_success(f"Model '{model_key}' downloaded successfully to: {model_dir}")
    return True


def list_models() -> None:
    """Display available models"""
    print_header("Available FireRed-VAD PTH Models")
    
    print(f"{Colors.BOLD}Repository:{Colors.ENDC} https://huggingface.co/{HUGGINGFACE_REPO}\n")
    print(f"{Colors.BOLD}Format:{Colors.ENDC} PyTorch .pth.tar (original models)\n")
    
    for model_key, model_info in MODELS.items():
        print(f"{Colors.BOLD}{Colors.OKCYAN}[{model_key}]{Colors.ENDC}")
        print(f"  Name:        {model_info['name']}")
        print(f"  Size:        ~{model_info['size_mb']} MB")
        print(f"  Description: {model_info['description']}")
        print(f"  Files:       {model_info['model_file']}, {model_info['cmvn_file']}")
        print()


def download_all_models(output_dir: Path, force: bool = False) -> None:
    """
    Download all available models
    
    Args:
        output_dir: Directory to save models
        force: Force re-download even if files exist
    """
    print_header("Downloading All FireRed-VAD PTH Models")
    
    total_models = len(MODELS)
    successful = 0
    failed = []
    
    for idx, model_key in enumerate(MODELS.keys(), 1):
        print(f"\n{Colors.BOLD}[{idx}/{total_models}] Downloading {model_key}...{Colors.ENDC}")
        
        if download_model(model_key, output_dir, force):
            successful += 1
        else:
            failed.append(model_key)
    
    # Summary
    print_header("Download Summary")
    print(f"{Colors.BOLD}Total models:{Colors.ENDC} {total_models}")
    print(f"{Colors.OKGREEN}Successful:{Colors.ENDC} {successful}")
    
    if failed:
        print(f"{Colors.FAIL}Failed:{Colors.ENDC} {len(failed)}")
        for model_key in failed:
            print(f"  - {model_key}")
    
    if successful == total_models:
        print_success("\nAll models downloaded successfully!")
        print_info(f"Models saved to: {output_dir.absolute()}")
        print_info("\nNext step: Run convert_all_models.bat to convert PTH to GGUF")
    elif successful > 0:
        print_warning("\nSome models downloaded successfully, but some failed")
    else:
        print_error("\nAll downloads failed")
        sys.exit(1)


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FireRed-VAD PTH Model Downloader - Download original PyTorch models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_pth_models.py                     # Download all models
  python download_pth_models.py --model vad         # Download VAD model only
  python download_pth_models.py --model stream-vad  # Download Streaming VAD model
  python download_pth_models.py --list              # List available models
  python download_pth_models.py --output ./models   # Custom output directory
  python download_pth_models.py --force             # Force re-download existing files

Available models: vad, stream-vad, aed

After downloading, use convert_all_models.bat to convert PTH to GGUF format.
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=list(MODELS.keys()),
        help='Download specific model (default: download all)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models and exit'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for models (default: pth_models in project root)'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download even if file exists'
    )
    
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    
    args = parser.parse_args()
    
    # Disable colors if requested or on Windows
    if args.no_color or os.name == 'nt':
        Colors.disable()
    
    # List models and exit
    if args.list:
        list_models()
        return
    
    output_dir = Path(args.output).resolve()
    
    # Download specific model or all models
    if args.model:
        print_header(f"FireRed-VAD PTH Model Downloader")
        if download_model(args.model, output_dir, args.force):
            print_success(f"\nModel saved to: {output_dir.absolute()}")
            print_info("\nNext step: Run convert_all_models.bat to convert PTH to GGUF")
        else:
            print_error("\nDownload failed")
            sys.exit(1)
    else:
        download_all_models(output_dir, args.force)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"\nFatal error: {str(e)}")
        sys.exit(1)
