#!/usr/bin/env python3
"""
Example usage of AWS Data Downloader

This script demonstrates how to use the AWS data downloader programmatically.
"""

import os
from aws_data_downloader import AWSDataDownloader


def main():
    """Example usage of the AWS data downloader."""
    
    # Get credentials from environment variables (recommended for security)
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    
    if not access_key or not secret_key:
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        print("Or modify this script to use your credentials directly")
        return
    
    # Option 1: Use with configuration file
    print("Option 1: Using configuration file")
    downloader = AWSDataDownloader('aws_downloader_config.yaml')
    
    # Option 2: Use with custom configuration
    print("\nOption 2: Using custom configuration")
    custom_config = {
        'download_dir': 'data',
        'parallel_downloads': 5,
        'file_extensions': ['.png', '.jpg', '.json'],
        'dry_run': True,  # Set to False for actual download
        'max_file_size_mb': 500,
        'buckets': [],  # Empty means all accessible buckets
        'prefixes': ['coryell/', 'road-distress/']  # Only download these prefixes
    }
    
    downloader_custom = AWSDataDownloader()
    downloader_custom.config.update(custom_config)
    
    # Run the download
    try:
        print("Starting download process...")
        downloader_custom.run_full_download(access_key, secret_key, region)
        print("Download completed successfully!")
        
    except Exception as e:
        print(f"Download failed: {e}")


if __name__ == '__main__':
    main() 