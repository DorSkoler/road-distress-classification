# AWS Data Downloader for Road Distress Classification

A robust, production-ready Python script for downloading data from AWS S3 buckets with advanced features including parallel downloads, resumable downloads, progress tracking, and comprehensive error handling.

## Features

- ✅ **Parallel Downloads**: Download multiple files concurrently for maximum speed
- ✅ **Resume Capability**: Automatically resume interrupted downloads
- ✅ **Progress Tracking**: Real-time progress bars and statistics
- ✅ **File Verification**: Verify downloaded files using checksums
- ✅ **Robust Error Handling**: Automatic retries with exponential backoff
- ✅ **Flexible Configuration**: YAML/JSON configuration files
- ✅ **Dry Run Mode**: Preview what would be downloaded without downloading
- ✅ **Smart Filtering**: Filter by file extensions, size, and patterns
- ✅ **Directory Structure**: Preserve S3 directory structure locally
- ✅ **Comprehensive Logging**: Detailed logs for debugging and monitoring

## Installation

1. **Install dependencies**:
```bash
pip install -r aws_downloader_requirements.txt
```

2. **Set up AWS credentials** (choose one method):

   **Option A: Environment variables (recommended)**:
   ```bash
   export AWS_ACCESS_KEY_ID="your_access_key"
   export AWS_SECRET_ACCESS_KEY="your_secret_key"
   export AWS_DEFAULT_REGION="us-east-1"
   ```

   **Option B: AWS CLI configuration**:
   ```bash
   aws configure
   ```

   **Option C: Direct in script** (not recommended for production):
   ```python
   access_key = "your_access_key"
   secret_key = "your_secret_key"
   ```

## Quick Start

### Command Line Usage

1. **Create default configuration**:
```bash
python aws_data_downloader.py --create-config
```

2. **Test with dry run**:
```bash
python aws_data_downloader.py \
    --access-key "YOUR_ACCESS_KEY" \
    --secret-key "YOUR_SECRET_KEY" \
    --dry-run
```

3. **Download all accessible data**:
```bash
python aws_data_downloader.py \
    --access-key "YOUR_ACCESS_KEY" \
    --secret-key "YOUR_SECRET_KEY" \
    --download-dir "data"
```

4. **Download specific buckets**:
```bash
python aws_data_downloader.py \
    --access-key "YOUR_ACCESS_KEY" \
    --secret-key "YOUR_SECRET_KEY" \
    --buckets "bucket1" "bucket2" \
    --parallel 20
```

### Programmatic Usage

```python
from aws_data_downloader import AWSDataDownloader

# Create downloader with configuration
downloader = AWSDataDownloader('aws_downloader_config.yaml')

# Run download
downloader.run_full_download(
    access_key="your_access_key",
    secret_key="your_secret_key",
    region="us-east-1"
)
```

## Configuration

Edit `aws_downloader_config.yaml` to customize download behavior:

```yaml
# Download settings
download_dir: "data"
parallel_downloads: 10
overwrite_existing: false
max_file_size_mb: 1000

# File filtering
file_extensions:
  - ".png"
  - ".jpg"
  - ".json"

# Specific buckets/prefixes
buckets: ["my-road-data-bucket"]
prefixes: ["coryell/", "road-distress/"]

# Performance tuning
retry_attempts: 3
verify_checksums: true
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--access-key` | AWS access key ID | `--access-key "AKIAIOSFODNN7EXAMPLE"` |
| `--secret-key` | AWS secret access key | `--secret-key "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"` |
| `--region` | AWS region | `--region "us-west-2"` |
| `--config` | Configuration file path | `--config "my_config.yaml"` |
| `--download-dir` | Download directory | `--download-dir "road_data"` |
| `--parallel` | Number of parallel downloads | `--parallel 20` |
| `--buckets` | Specific buckets | `--buckets "bucket1" "bucket2"` |
| `--prefixes` | Specific prefixes | `--prefixes "coryell/" "images/"` |
| `--dry-run` | Preview without downloading | `--dry-run` |
| `--create-config` | Create default config file | `--create-config` |

## Advanced Usage Examples

### 1. Download Only Road Images
```bash
python aws_data_downloader.py \
    --access-key "YOUR_KEY" \
    --secret-key "YOUR_SECRET" \
    --prefixes "coryell/" \
    --download-dir "road_images" \
    --parallel 15
```

### 2. Large Dataset Download with Verification
```yaml
# config.yaml
download_dir: "large_dataset"
parallel_downloads: 25
verify_checksums: true
retry_attempts: 5
max_file_size_mb: 2000
file_extensions: [".png", ".jpg", ".tif", ".json"]
```

```bash
python aws_data_downloader.py \
    --config "config.yaml" \
    --access-key "YOUR_KEY" \
    --secret-key "YOUR_SECRET"
```

### 3. Resume Interrupted Download
The script automatically resumes interrupted downloads. Simply run the same command again:

```bash
# This will resume from where it left off
python aws_data_downloader.py \
    --access-key "YOUR_KEY" \
    --secret-key "YOUR_SECRET"
```

### 4. Download with Size Limits
```bash
python aws_data_downloader.py \
    --access-key "YOUR_KEY" \
    --secret-key "YOUR_SECRET" \
    --config "config.yaml"
```

Where `config.yaml` contains:
```yaml
max_file_size_mb: 100  # Skip files larger than 100MB
file_extensions: [".png", ".jpg"]  # Only images
```

## Performance Optimization

### Recommended Settings by Use Case

**Fast Network (>100 Mbps)**:
```yaml
parallel_downloads: 20-30
chunk_size: 16384
retry_attempts: 5
```

**Slow Network (<10 Mbps)**:
```yaml
parallel_downloads: 3-5
chunk_size: 4096
retry_attempts: 3
```

**Large Files (>100MB each)**:
```yaml
parallel_downloads: 5-10
chunk_size: 32768
max_file_size_mb: 5000
```

**Many Small Files**:
```yaml
parallel_downloads: 15-25
chunk_size: 8192
verify_checksums: false  # For speed
```

## Monitoring and Logging

The script provides comprehensive logging and monitoring:

### Console Output
```
2024-01-15 10:30:15 - INFO - AWS credentials validated successfully
2024-01-15 10:30:16 - INFO - Discovered 3 accessible buckets
2024-01-15 10:30:20 - INFO - Found 1,234 objects to download
Downloading: 45%|████▌     | 556/1234 [02:15<02:45, 4.1files/s, Success=550, Failed=6, Speed=15.2 MB/s]
```

### Log Files
- `aws_downloader.log`: Detailed operation logs
- `download_resume.json`: Resume data for interrupted downloads

### Summary Report
```
============================================================
DOWNLOAD SUMMARY
============================================================
Total files:      1,234
Downloaded:       1,200
Skipped:          28
Failed:           6
Total size:       15.2 GB
Downloaded size:  14.8 GB
Duration:         1,245.3 seconds
Average speed:    12.1 MB/s
Success rate:     97.2%
```

## Error Handling and Troubleshooting

### Common Issues

**1. Credentials Error**
```
Error: AWS credentials not found or invalid
```
**Solution**: Verify your AWS credentials are correct and have S3 access permissions.

**2. Permission Denied**
```
Error: Access Denied for bucket 'my-bucket'
```
**Solution**: Ensure your AWS user has `s3:GetObject` and `s3:ListBucket` permissions.

**3. Network Timeout**
```
Error: Connection timeout
```
**Solution**: Reduce `parallel_downloads` and increase `retry_attempts`.

**4. Disk Space**
```
Error: No space left on device
```
**Solution**: Free up disk space or change `download_dir` to a location with more space.

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

Or use verbose mode:
```bash
python aws_data_downloader.py --verbose
```

## Security Best Practices

1. **Use Environment Variables**: Store credentials in environment variables, not in code
2. **IAM Roles**: Use IAM roles instead of access keys when running on EC2
3. **Least Privilege**: Grant only necessary S3 permissions
4. **Rotate Keys**: Regularly rotate access keys
5. **Monitor Access**: Use CloudTrail to monitor S3 access

### Example IAM Policy
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name",
                "arn:aws:s3:::your-bucket-name/*"
            ]
        }
    ]
}
```

## File Structure After Download

```
data/
├── bucket1/
│   ├── coryell/
│   │   ├── Co Rd 4235/
│   │   │   ├── img/
│   │   │   │   ├── 000_31.708136_-97.693460.png
│   │   │   │   └── 001_31.708228_-97.693279.png
│   │   │   └── ann/
│   │   │       └── annotations.json
│   │   └── Co Rd 360/
│   └── other-data/
├── bucket2/
│   └── road-distress/
└── download_resume.json
```

## Integration with Existing Project

The downloader integrates seamlessly with your road distress classification project:

```python
# After downloading data
from aws_data_downloader import AWSDataDownloader

# Download data
downloader = AWSDataDownloader()
downloader.run_full_download(access_key, secret_key)

# Your existing code can now use the downloaded data
from your_project import train_model
train_model(data_dir="data/coryell")
```

## Contributing

To contribute to this downloader:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This script is part of the road distress classification project and follows the same license terms.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the log files for detailed error messages
3. Create an issue with detailed information about your setup and the error 