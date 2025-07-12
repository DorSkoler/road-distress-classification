#!/usr/bin/env python3
"""
AWS Data Downloader for Road Distress Classification Project

This script downloads all available data from AWS S3 buckets using provided credentials.
It includes robust error handling, progress tracking, parallel downloads, and resumable downloads.
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
from tqdm import tqdm
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aws_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DownloadStats:
    """Statistics for download operations"""
    total_files: int = 0
    downloaded_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    total_size: int = 0
    downloaded_size: int = 0
    start_time: float = 0
    end_time: float = 0
    
    def get_duration(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else time.time() - self.start_time
    
    def get_speed_mbps(self) -> float:
        duration = self.get_duration()
        if duration > 0:
            return (self.downloaded_size / (1024 * 1024)) / duration
        return 0.0


class AWSDataDownloader:
    """Comprehensive AWS S3 data downloader with advanced features"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the AWS data downloader.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config = self._load_config(config_file)
        self.s3_client = None
        self.stats = DownloadStats()
        self.downloaded_files: Set[str] = set()
        self.failed_files: Set[str] = set()
        
        # Setup paths
        self.base_download_dir = Path(self.config.get('download_dir', 'data'))
        self.base_download_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume capability
        self.resume_file = Path('download_resume.json')
        self.load_resume_data()
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'download_dir': 'data',
            'parallel_downloads': 10,
            'chunk_size': 8192,
            'retry_attempts': 3,
            'retry_delay': 1,
            'file_extensions': ['.png', '.jpg', '.jpeg', '.json', '.txt', '.csv', '.tif', '.tiff'],
            'excluded_patterns': ['.DS_Store', 'Thumbs.db', '__pycache__'],
            'verify_checksums': True,
            'create_directory_structure': True,
            'overwrite_existing': False,
            'buckets': [],  # Will be discovered automatically
            'prefixes': [],  # Will download all prefixes if empty
            'max_file_size_mb': 1000,  # Skip files larger than 1GB
            'dry_run': False
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)
                default_config.update(file_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def setup_aws_client(self, access_key: str, secret_key: str, region: str = 'us-east-1'):
        """Setup AWS S3 client with credentials.
        
        Args:
            access_key: AWS access key ID
            secret_key: AWS secret access key
            region: AWS region (default: us-east-1)
        """
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )
            
            # Test credentials
            self.s3_client.list_buckets()
            logger.info("AWS credentials validated successfully")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found or invalid")
            raise
        except ClientError as e:
            logger.error(f"AWS client error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error setting up AWS client: {e}")
            raise
    
    def discover_buckets(self) -> List[str]:
        """Discover all accessible S3 buckets.
        
        Returns:
            List of bucket names
        """
        if not self.s3_client:
            raise RuntimeError("AWS client not initialized")
        
        try:
            response = self.s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            
            # Filter buckets based on configuration
            if self.config.get('buckets'):
                buckets = [b for b in buckets if b in self.config['buckets']]
            
            logger.info(f"Discovered {len(buckets)} accessible buckets")
            return buckets
            
        except ClientError as e:
            logger.error(f"Error discovering buckets: {e}")
            return []
    
    def list_bucket_objects(self, bucket_name: str, prefix: str = '') -> List[Dict]:
        """List all objects in a bucket with optional prefix.
        
        Args:
            bucket_name: Name of the S3 bucket
            prefix: Optional prefix to filter objects
            
        Returns:
            List of object metadata dictionaries
        """
        if not self.s3_client:
            raise RuntimeError("AWS client not initialized")
        
        objects = []
        continuation_token = None
        
        try:
            while True:
                list_params = {
                    'Bucket': bucket_name,
                    'Prefix': prefix,
                    'MaxKeys': 1000
                }
                
                if continuation_token:
                    list_params['ContinuationToken'] = continuation_token
                
                response = self.s3_client.list_objects_v2(**list_params)
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        # Filter by file extension
                        if self._should_download_file(obj['Key']):
                            objects.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'],
                                'etag': obj['ETag'].strip('"'),
                                'bucket': bucket_name
                            })
                
                if not response.get('IsTruncated', False):
                    break
                
                continuation_token = response.get('NextContinuationToken')
                
        except ClientError as e:
            logger.error(f"Error listing objects in bucket {bucket_name}: {e}")
            return []
        
        logger.info(f"Found {len(objects)} objects in bucket {bucket_name} with prefix '{prefix}'")
        return objects
    
    def _should_download_file(self, key: str) -> bool:
        """Check if a file should be downloaded based on configuration.
        
        Args:
            key: S3 object key
            
        Returns:
            True if file should be downloaded
        """
        # Check file extension
        if self.config['file_extensions']:
            if not any(key.lower().endswith(ext.lower()) for ext in self.config['file_extensions']):
                return False
        
        # Check excluded patterns
        for pattern in self.config['excluded_patterns']:
            if pattern in key:
                return False
        
        return True
    
    def download_file(self, bucket: str, key: str, local_path: Path, 
                     file_size: int, etag: str) -> bool:
        """Download a single file from S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            local_path: Local file path
            file_size: Expected file size
            etag: Expected ETag for verification
            
        Returns:
            True if download was successful
        """
        if not self.s3_client:
            raise RuntimeError("AWS client not initialized")
        
        # Check if file already exists and is valid
        if local_path.exists() and not self.config['overwrite_existing']:
            if self._verify_file(local_path, file_size, etag):
                logger.debug(f"File already exists and is valid: {local_path}")
                return True
        
        # Check file size limit
        max_size = self.config['max_file_size_mb'] * 1024 * 1024
        if file_size > max_size:
            logger.warning(f"Skipping large file ({file_size / (1024*1024):.1f}MB): {key}")
            return False
        
        # Create directory if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with retries
        for attempt in range(self.config['retry_attempts']):
            try:
                if self.config['dry_run']:
                    logger.info(f"[DRY RUN] Would download: {bucket}/{key} -> {local_path}")
                    return True
                
                # Download file
                with open(local_path, 'wb') as f:
                    self.s3_client.download_fileobj(bucket, key, f)
                
                # Verify download
                if self._verify_file(local_path, file_size, etag):
                    logger.debug(f"Successfully downloaded: {local_path}")
                    return True
                else:
                    logger.warning(f"Download verification failed: {local_path}")
                    local_path.unlink(missing_ok=True)
                    
            except ClientError as e:
                logger.error(f"Download error (attempt {attempt + 1}): {e}")
                if attempt < self.config['retry_attempts'] - 1:
                    time.sleep(self.config['retry_delay'] * (2 ** attempt))
                else:
                    return False
            except Exception as e:
                logger.error(f"Unexpected download error: {e}")
                return False
        
        return False
    
    def _verify_file(self, file_path: Path, expected_size: int, expected_etag: str) -> bool:
        """Verify downloaded file integrity.
        
        Args:
            file_path: Path to downloaded file
            expected_size: Expected file size
            expected_etag: Expected ETag
            
        Returns:
            True if file is valid
        """
        if not file_path.exists():
            return False
        
        # Check file size
        actual_size = file_path.stat().st_size
        if actual_size != expected_size:
            logger.warning(f"Size mismatch: expected {expected_size}, got {actual_size}")
            return False
        
        # Check ETag if verification is enabled
        if self.config['verify_checksums']:
            try:
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash != expected_etag:
                    logger.warning(f"ETag mismatch: expected {expected_etag}, got {file_hash}")
                    return False
            except Exception as e:
                logger.warning(f"Failed to verify checksum: {e}")
                return False
        
        return True
    
    def download_objects_parallel(self, objects: List[Dict]) -> None:
        """Download multiple objects in parallel.
        
        Args:
            objects: List of object metadata dictionaries
        """
        self.stats.total_files = len(objects)
        self.stats.total_size = sum(obj['size'] for obj in objects)
        self.stats.start_time = time.time()
        
        logger.info(f"Starting download of {self.stats.total_files} files "
                   f"({self.stats.total_size / (1024*1024):.1f} MB)")
        
        with ThreadPoolExecutor(max_workers=self.config['parallel_downloads']) as executor:
            # Submit all download tasks
            future_to_object = {}
            for obj in objects:
                local_path = self._get_local_path(obj['bucket'], obj['key'])
                future = executor.submit(
                    self.download_file,
                    obj['bucket'],
                    obj['key'],
                    local_path,
                    obj['size'],
                    obj['etag']
                )
                future_to_object[future] = obj
            
            # Process completed downloads with progress bar
            with tqdm(total=self.stats.total_files, desc="Downloading", unit="files") as pbar:
                for future in as_completed(future_to_object):
                    obj = future_to_object[future]
                    try:
                        success = future.result()
                        if success:
                            self.stats.downloaded_files += 1
                            self.stats.downloaded_size += obj['size']
                            self.downloaded_files.add(f"{obj['bucket']}/{obj['key']}")
                        else:
                            self.stats.failed_files += 1
                            self.failed_files.add(f"{obj['bucket']}/{obj['key']}")
                    except Exception as e:
                        logger.error(f"Download task failed: {e}")
                        self.stats.failed_files += 1
                        self.failed_files.add(f"{obj['bucket']}/{obj['key']}")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': self.stats.downloaded_files,
                        'Failed': self.stats.failed_files,
                        'Speed': f"{self.stats.get_speed_mbps():.1f} MB/s"
                    })
        
        self.stats.end_time = time.time()
        self.save_resume_data()
    
    def _get_local_path(self, bucket: str, key: str) -> Path:
        """Get local file path for S3 object.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            Local file path
        """
        if self.config['create_directory_structure']:
            return self.base_download_dir / bucket / key
        else:
            return self.base_download_dir / Path(key).name
    
    def save_resume_data(self) -> None:
        """Save resume data to file."""
        resume_data = {
            'downloaded_files': list(self.downloaded_files),
            'failed_files': list(self.failed_files),
            'timestamp': datetime.now().isoformat(),
            'stats': {
                'total_files': self.stats.total_files,
                'downloaded_files': self.stats.downloaded_files,
                'failed_files': self.stats.failed_files,
                'total_size': self.stats.total_size,
                'downloaded_size': self.stats.downloaded_size
            }
        }
        
        try:
            with open(self.resume_file, 'w') as f:
                json.dump(resume_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save resume data: {e}")
    
    def load_resume_data(self) -> None:
        """Load resume data from file."""
        if not self.resume_file.exists():
            return
        
        try:
            with open(self.resume_file, 'r') as f:
                resume_data = json.load(f)
            
            self.downloaded_files = set(resume_data.get('downloaded_files', []))
            self.failed_files = set(resume_data.get('failed_files', []))
            
            logger.info(f"Loaded resume data: {len(self.downloaded_files)} downloaded, "
                       f"{len(self.failed_files)} failed")
            
        except Exception as e:
            logger.warning(f"Failed to load resume data: {e}")
    
    def print_summary(self) -> None:
        """Print download summary."""
        duration = self.stats.get_duration()
        speed = self.stats.get_speed_mbps()
        
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total files:      {self.stats.total_files}")
        print(f"Downloaded:       {self.stats.downloaded_files}")
        print(f"Skipped:          {self.stats.skipped_files}")
        print(f"Failed:           {self.stats.failed_files}")
        print(f"Total size:       {self.stats.total_size / (1024*1024):.1f} MB")
        print(f"Downloaded size:  {self.stats.downloaded_size / (1024*1024):.1f} MB")
        print(f"Duration:         {duration:.1f} seconds")
        print(f"Average speed:    {speed:.1f} MB/s")
        print(f"Success rate:     {(self.stats.downloaded_files / max(1, self.stats.total_files)) * 100:.1f}%")
        
        if self.failed_files:
            print(f"\nFailed downloads:")
            for failed_file in sorted(self.failed_files):
                print(f"  - {failed_file}")
    
    def run_full_download(self, access_key: str, secret_key: str, region: str = 'us-east-1') -> None:
        """Run complete download process.
        
        Args:
            access_key: AWS access key ID
            secret_key: AWS secret access key
            region: AWS region
        """
        try:
            # Setup AWS client
            self.setup_aws_client(access_key, secret_key, region)
            
            # Discover buckets
            buckets = self.discover_buckets()
            if not buckets:
                logger.error("No accessible buckets found")
                return
            
            # Collect all objects to download
            all_objects = []
            for bucket in buckets:
                logger.info(f"Scanning bucket: {bucket}")
                
                # Use configured prefixes or scan all
                prefixes = self.config.get('prefixes', [''])
                for prefix in prefixes:
                    objects = self.list_bucket_objects(bucket, prefix)
                    all_objects.extend(objects)
            
            if not all_objects:
                logger.warning("No objects found to download")
                return
            
            # Filter out already downloaded files
            if not self.config['overwrite_existing']:
                all_objects = [obj for obj in all_objects 
                             if f"{obj['bucket']}/{obj['key']}" not in self.downloaded_files]
            
            logger.info(f"Found {len(all_objects)} objects to download")
            
            # Start parallel download
            if all_objects:
                self.download_objects_parallel(all_objects)
            
            # Print summary
            self.print_summary()
            
        except Exception as e:
            logger.error(f"Download process failed: {e}")
            raise


def create_default_config() -> None:
    """Create default configuration file."""
    config = {
        "download_dir": "data",
        "parallel_downloads": 10,
        "chunk_size": 8192,
        "retry_attempts": 3,
        "retry_delay": 1,
        "file_extensions": [".png", ".jpg", ".jpeg", ".json", ".txt", ".csv", ".tif", ".tiff"],
        "excluded_patterns": [".DS_Store", "Thumbs.db", "__pycache__"],
        "verify_checksums": True,
        "create_directory_structure": True,
        "overwrite_existing": False,
        "max_file_size_mb": 1000,
        "dry_run": False,
        "buckets": [],
        "prefixes": ['coryell/', 'road-distress/']
    }
    
    with open('aws_downloader_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("Created default configuration file: aws_downloader_config.yaml")
    print("Edit this file to customize download settings.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AWS S3 Data Downloader')
    parser.add_argument('--access-key', required=True, help='AWS access key ID')
    parser.add_argument('--secret-key', required=True, help='AWS secret access key')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--create-config', action='store_true', 
                       help='Create default configuration file')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be downloaded without downloading')
    parser.add_argument('--download-dir', help='Override download directory')
    parser.add_argument('--parallel', type=int, help='Number of parallel downloads')
    parser.add_argument('--buckets', nargs='+', help='Specific buckets to download from')
    parser.add_argument('--prefixes', nargs='+', help='Specific prefixes to download')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config()
        return
    
    # Create downloader
    downloader = AWSDataDownloader(args.config)
    
    # Override config with command line arguments
    if args.dry_run:
        downloader.config['dry_run'] = True
    if args.download_dir:
        downloader.config['download_dir'] = args.download_dir
        downloader.base_download_dir = Path(args.download_dir)
        downloader.base_download_dir.mkdir(parents=True, exist_ok=True)
    if args.parallel:
        downloader.config['parallel_downloads'] = args.parallel
    if args.buckets:
        downloader.config['buckets'] = args.buckets
    if args.prefixes:
        downloader.config['prefixes'] = args.prefixes
    
    # Run download
    try:
        downloader.run_full_download(args.access_key, args.secret_key, args.region)
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        downloader.print_summary()
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 