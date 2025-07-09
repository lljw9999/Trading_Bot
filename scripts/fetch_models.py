#!/usr/bin/env python3
"""
Hugging Face Model Fetch Utility

Downloads transformer models from Hugging Face Hub, caches locally,
and optionally syncs ONNX exports to S3 model bucket.

Usage:
    python scripts/fetch_models.py tlob_tiny                    # Download single model
    python scripts/fetch_models.py --all                       # Download all models  
    python scripts/fetch_models.py --upload-s3 tlob_tiny       # Download + upload to S3
    python scripts/fetch_models.py --list                      # List available models
"""

import os
import sys
import argparse
import logging
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    HF_AVAILABLE = True
except ImportError:
    print("⚠️  huggingface_hub not installed. Run: pip install huggingface-hub")
    HF_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Model Registry - Maps our internal names to HF repos and key files
MODELS = {
    'tlob_tiny': {
        'repo_id': 'LeonardoBerti00/TLOB-FI-2010',
        'key_files': ['tlob_tiny.pt', 'config.json'],
        'description': 'TLOB Transformer for FI-2010 dataset',
        'size_mb': 45
    },
    'patchtst_small': {
        'repo_id': 'ibm-granite/granite-timeseries-patchtst',
        'key_files': ['pytorch_model.bin', 'config.json'],
        'description': 'Granite PatchTST for time series forecasting',
        'size_mb': 120
    },
    'timesnet_base': {
        'repo_id': 'timeseriesAI/TimesNet',
        'key_files': ['pytorch_model.bin', 'config.json'],
        'description': 'TimesNet encoder/decoder for time series',
        'size_mb': 200
    },
    'mambats_small': {
        'repo_id': 'state-spaces/mamba-timeseries-small',
        'key_files': ['pytorch_model.bin', 'config.json'],
        'description': 'Mamba architecture for time series (ONNX compatible)',
        'size_mb': 80
    },
    'tide_small': {
        'repo_id': 'ours/tide-small',  # Custom implementation
        'key_files': ['model.pt', 'config.json'],
        'description': 'TiDE Dense Encoder (custom implementation)',
        'size_mb': 25,
        'custom': True
    },
    'chronos_bolt_base': {
        'repo_id': 'amazon/chronos-bolt-base',
        'key_files': ['pytorch_model.bin', 'config.json'],
        'description': 'Amazon Chronos Bolt (pre-quantized INT8)',
        'size_mb': 150
    }
}

# Paths
CACHE_DIR = Path.home() / '.cache' / 'hf_models'
MODELS_DIR = Path(__file__).parent.parent / 'models'
S3_BUCKET = 's3://trading-models'

class ModelFetcher:
    """Handles downloading and managing HF models."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def get_model_cache_path(self, model_name: str) -> Path:
        """Get cache directory for a specific model."""
        return self.cache_dir / model_name
    
    def get_file_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get cached model information."""
        cache_path = self.get_model_cache_path(model_name)
        info_file = cache_path / 'model_info.json'
        
        if info_file.exists():
            with open(info_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_model_info(self, model_name: str, info: Dict):
        """Save model information to cache."""
        cache_path = self.get_model_cache_path(model_name)
        cache_path.mkdir(parents=True, exist_ok=True)
        info_file = cache_path / 'model_info.json'
        
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """
        Download model from Hugging Face Hub.
        
        Args:
            model_name: Model name from MODELS registry
            force: Force re-download even if cached
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in MODELS:
            log.error(f"❌ Unknown model: {model_name}")
            return False
        
        model_config = MODELS[model_name]
        cache_path = self.get_model_cache_path(model_name)
        
        # Check if model is custom (not on HF Hub)
        if model_config.get('custom', False):
            log.warning(f"⚠️  {model_name} is a custom model - implement locally")
            return False
        
        # Check if already cached and not forcing
        if not force and cache_path.exists() and any(cache_path.iterdir()):
            log.info(f"✅ {model_name} already cached at {cache_path}")
            return True
        
        try:
            log.info(f"📦 Downloading {model_name} from {model_config['repo_id']}...")
            
            # Download using HF snapshot_download
            snapshot_path = snapshot_download(
                repo_id=model_config['repo_id'],
                cache_dir=str(cache_path),
                local_dir=str(cache_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            # Verify key files exist
            missing_files = []
            total_size = 0
            file_info = {}
            
            for key_file in model_config['key_files']:
                file_path = cache_path / key_file
                if file_path.exists():
                    size = file_path.stat().st_size
                    total_size += size
                    file_info[key_file] = {
                        'size': size,
                        'sha256': self.get_file_sha256(file_path)
                    }
                else:
                    missing_files.append(key_file)
            
            if missing_files:
                log.warning(f"⚠️  Missing key files for {model_name}: {missing_files}")
            
            # Save model metadata
            model_info = {
                'model_name': model_name,
                'repo_id': model_config['repo_id'],
                'description': model_config['description'],
                'download_date': datetime.now().isoformat(),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'files': file_info,
                'cache_path': str(cache_path)
            }
            
            self.save_model_info(model_name, model_info)
            
            log.info(f"✅ Downloaded {model_name}: {total_size / (1024*1024):.1f} MB")
            log.info(f"   Cached at: {cache_path}")
            
            return True
            
        except Exception as e:
            log.error(f"❌ Failed to download {model_name}: {e}")
            return False
    
    def list_models(self) -> List[Dict]:
        """List all available models with their status."""
        models_status = []
        
        for model_name, config in MODELS.items():
            cache_path = self.get_model_cache_path(model_name)
            model_info = self.get_model_info(model_name)
            
            status = {
                'name': model_name,
                'repo_id': config['repo_id'],
                'description': config['description'],
                'estimated_size_mb': config['size_mb'],
                'custom': config.get('custom', False),
                'cached': cache_path.exists() and any(cache_path.iterdir()),
                'cache_path': str(cache_path) if cache_path.exists() else None,
                'download_date': model_info.get('download_date'),
                'actual_size_mb': model_info.get('total_size_mb', 0)
            }
            
            models_status.append(status)
        
        return models_status
    
    def upload_onnx_to_s3(self, model_name: str) -> bool:
        """
        Upload ONNX files to S3 model bucket.
        
        Args:
            model_name: Model name to upload
            
        Returns:
            True if successful, False otherwise
        """
        # Check for ONNX files in models directory
        onnx_files = list(self.models_dir.glob(f"{model_name}*.onnx"))
        
        if not onnx_files:
            log.warning(f"⚠️  No ONNX files found for {model_name} in {self.models_dir}")
            return False
        
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            s3_client = boto3.client('s3')
            bucket_name = S3_BUCKET.replace('s3://', '')
            
            uploaded_files = []
            for onnx_file in onnx_files:
                s3_key = f"{model_name}/{onnx_file.name}"
                
                log.info(f"📤 Uploading {onnx_file.name} to {S3_BUCKET}/{s3_key}...")
                
                s3_client.upload_file(
                    str(onnx_file),
                    bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': 'application/octet-stream'}
                )
                
                uploaded_files.append(s3_key)
                log.info(f"✅ Uploaded: s3://{bucket_name}/{s3_key}")
            
            log.info(f"🎉 Successfully uploaded {len(uploaded_files)} ONNX files for {model_name}")
            return True
            
        except ImportError:
            log.error("❌ boto3 not installed. Run: pip install boto3")
            return False
        except ClientError as e:
            log.error(f"❌ S3 upload failed: {e}")
            return False
        except Exception as e:
            log.error(f"❌ Unexpected error during S3 upload: {e}")
            return False


def print_models_table(models_status: List[Dict]):
    """Print a formatted table of model status."""
    print("\n📚 Model Registry Status:")
    print("=" * 100)
    print(f"{'Name':<20} {'Repo ID':<35} {'Cached':<8} {'Size (MB)':<10} {'Downloaded':<12}")
    print("=" * 100)
    
    total_size = 0
    cached_count = 0
    
    for model in models_status:
        cached_icon = "✅" if model['cached'] else "❌"
        size_str = f"{model['actual_size_mb']:.1f}" if model['cached'] else f"~{model['estimated_size_mb']}"
        download_date = model['download_date'][:10] if model['download_date'] else "Never"
        custom_marker = " (custom)" if model['custom'] else ""
        
        print(f"{model['name']:<20} {model['repo_id']:<35} {cached_icon:<8} {size_str:<10} {download_date:<12}")
        
        if model['cached']:
            total_size += model['actual_size_mb']
            cached_count += 1
    
    print("=" * 100)
    print(f"Summary: {cached_count}/{len(models_status)} models cached, {total_size:.1f} MB total")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hugging Face Model Fetch Utility")
    parser.add_argument('model_name', nargs='?', help='Model name to download')
    parser.add_argument('--all', action='store_true', help='Download all models')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--upload-s3', action='store_true', help='Upload ONNX files to S3 after download')
    parser.add_argument('--force', action='store_true', help='Force re-download even if cached')
    parser.add_argument('--cache-dir', type=str, help='Custom cache directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not HF_AVAILABLE:
        log.error("❌ huggingface_hub not available. Install with: pip install huggingface-hub")
        sys.exit(1)
    
    # Initialize fetcher
    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
    fetcher = ModelFetcher(cache_dir)
    
    # Handle list command
    if args.list:
        models_status = fetcher.list_models()
        print_models_table(models_status)
        return
    
    # Handle download commands
    models_to_download = []
    
    if args.all:
        models_to_download = [name for name in MODELS.keys() if not MODELS[name].get('custom', False)]
        log.info(f"📦 Downloading all {len(models_to_download)} models...")
    elif args.model_name:
        if args.model_name not in MODELS:
            log.error(f"❌ Unknown model: {args.model_name}")
            log.info(f"Available models: {', '.join(MODELS.keys())}")
            sys.exit(1)
        models_to_download = [args.model_name]
    else:
        parser.print_help()
        return
    
    # Download models
    success_count = 0
    start_time = datetime.now()
    
    for model_name in models_to_download:
        log.info(f"\n🎯 Processing {model_name}...")
        
        if fetcher.download_model(model_name, force=args.force):
            success_count += 1
            
            # Upload to S3 if requested
            if args.upload_s3:
                fetcher.upload_onnx_to_s3(model_name)
        else:
            log.error(f"❌ Failed to download {model_name}")
    
    # Summary
    elapsed = datetime.now() - start_time
    log.info(f"\n🎉 Download Summary:")
    log.info(f"   ✅ Successfully downloaded: {success_count}/{len(models_to_download)} models")
    log.info(f"   ⏱️  Total time: {elapsed.total_seconds():.1f} seconds")
    
    if success_count > 0:
        log.info(f"   📁 Models cached in: {fetcher.cache_dir}")
        
        # Show final status
        models_status = fetcher.list_models()
        print_models_table(models_status)


if __name__ == '__main__':
    main() 