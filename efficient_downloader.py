#!/usr/bin/env python3
"""
Efficient AWS S3 downloader for v1/coryell/ data
Downloads while discovering objects and allows limiting
"""

import os
import boto3
from pathlib import Path
from tqdm import tqdm
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import time

class EfficientDownloader:
    def __init__(self, access_key, secret_key, bucket_name, prefix, download_dir="data", max_workers=10):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1'
        )
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.download_dir = Path(download_dir)
        self.max_workers = max_workers
        
        # Create download directory
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.discovered = 0
        self.downloaded = 0
        self.failed = 0
        self.skipped = 0
        
    def download_file(self, obj):
        """Download a single file"""
        key = obj['Key']
        size = obj['Size']
        
        # Create local path preserving S3 structure
        local_path = self.download_dir / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if file already exists and has same size
        if local_path.exists() and local_path.stat().st_size == size:
            self.skipped += 1
            return "skipped"
        
        try:
            # Download the file
            self.s3_client.download_file(self.bucket_name, key, str(local_path))
            self.downloaded += 1
            return "downloaded"
            
        except ClientError as e:
            print(f"\nError downloading {key}: {e}")
            self.failed += 1
            return "failed"
    
    def discover_and_download(self, max_files=None):
        """Discover objects and download them simultaneously"""
        print(f"Downloading from bucket: {self.bucket_name}")
        print(f"Prefix: {self.prefix}")
        print(f"Download directory: {self.download_dir}")
        if max_files:
            print(f"Maximum files: {max_files}")
        print("-" * 50)
        
        # Queue for objects to download
        download_queue = queue.Queue()
        discovery_complete = threading.Event()
        
        def discover_objects():
            """Discover objects and add them to download queue"""
            continuation_token = None
            
            while True:
                list_params = {
                    'Bucket': self.bucket_name,
                    'Prefix': self.prefix,
                    'MaxKeys': 1000
                }
                
                if continuation_token:
                    list_params['ContinuationToken'] = continuation_token
                
                try:
                    response = self.s3_client.list_objects_v2(**list_params)
                    
                    if 'Contents' in response:
                        for obj in response['Contents']:
                            self.discovered += 1
                            download_queue.put(obj)
                            
                            # Stop if we've reached the max files limit
                            if max_files and self.discovered >= max_files:
                                discovery_complete.set()
                                return
                    
                    if not response.get('IsTruncated', False):
                        break
                        
                    continuation_token = response.get('NextContinuationToken')
                    
                except ClientError as e:
                    print(f"Error listing objects: {e}")
                    break
            
            discovery_complete.set()
        
        # Start discovery in background
        discovery_thread = threading.Thread(target=discover_objects)
        discovery_thread.start()
        
        # Start downloading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            with tqdm(desc="Processing", unit="files") as pbar:
                while not discovery_complete.is_set() or not download_queue.empty():
                    # Submit download tasks
                    while len(futures) < self.max_workers and not download_queue.empty():
                        try:
                            obj = download_queue.get_nowait()
                            future = executor.submit(self.download_file, obj)
                            futures.append(future)
                        except queue.Empty:
                            break
                    
                    # Process completed downloads
                    completed_futures = []
                    for future in futures:
                        if future.done():
                            completed_futures.append(future)
                            try:
                                result = future.result()
                                pbar.set_postfix({
                                    "Discovered": self.discovered,
                                    "Downloaded": self.downloaded,
                                    "Failed": self.failed,
                                    "Skipped": self.skipped
                                })
                                pbar.update(1)
                            except Exception as e:
                                print(f"Download task failed: {e}")
                    
                    # Remove completed futures
                    for future in completed_futures:
                        futures.remove(future)
                    
                    # Small delay to prevent busy waiting
                    if not futures:
                        time.sleep(0.1)
        
        # Wait for discovery to complete
        discovery_thread.join()
        
        print(f"\nDownload complete!")
        print(f"Discovered: {self.discovered} files")
        print(f"Downloaded: {self.downloaded} files")
        print(f"Skipped: {self.skipped} files (already exist)")
        print(f"Failed: {self.failed} files")
        
        return self.downloaded > 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Efficient Coryell Road Distress Dataset Downloader')
    parser.add_argument('--access-key', required=True, help='AWS access key ID')
    parser.add_argument('--secret-key', required=True, help='AWS secret access key')
    parser.add_argument('--bucket', default='road-distress-datasets', help='S3 bucket name')
    parser.add_argument('--prefix', default='v1/coryell/', help='S3 prefix to download')
    parser.add_argument('--download-dir', default='data', help='Download directory')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to download')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel downloads (1-20)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode to ask for preferences')
    
    args = parser.parse_args()
    
    # Validate workers
    workers = max(1, min(20, args.workers))
    
    if args.interactive:
        # Interactive mode - ask user for preferences
        print("Coryell Road Distress Dataset Downloader")
        print("=" * 50)
        print("This bucket contains 27,000+ files (several GB)")
        print()
        
        try:
            max_files_input = input(f"Enter maximum files to download (current: {args.max_files or 'all'}): ").strip()
            max_files = int(max_files_input) if max_files_input else args.max_files
        except ValueError:
            max_files = args.max_files
        
        try:
            workers_input = input(f"Enter number of parallel downloads (1-20, current: {workers}): ").strip()
            workers = int(workers_input) if workers_input else workers
            workers = max(1, min(20, workers))
        except ValueError:
            pass
        
        download_dir = input(f"Enter download directory (current: {args.download_dir}): ").strip() or args.download_dir
    else:
        # Non-interactive mode - use command line arguments
        max_files = args.max_files
        download_dir = args.download_dir
        print("Coryell Road Distress Dataset Downloader")
        print("=" * 50)
        print(f"Bucket: {args.bucket}")
        print(f"Prefix: {args.prefix}")
        print(f"Download directory: {download_dir}")
        print(f"Max files: {max_files or 'all'}")
        print(f"Parallel workers: {workers}")
    
    print(f"\nStarting download with {workers} parallel workers...")
    
    downloader = EfficientDownloader(
        args.access_key, args.secret_key, args.bucket, args.prefix, download_dir, workers
    )
    
    try:
        success = downloader.discover_and_download(max_files)
        
        if success:
            print("\nDownload completed successfully!")
        else:
            print("\nDownload failed!")
            
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        print(f"Downloaded: {downloader.downloaded} files")
        print(f"Failed: {downloader.failed} files")

if __name__ == '__main__':
    main() 