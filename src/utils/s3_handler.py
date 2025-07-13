"""
Axolotl S3 Handler
Utility class for managing S3 operations with boto3
"""

import os
import boto3
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from urllib.parse import urlparse
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm


class S3Handler:
    """
    Handler for S3 operations with support for custom endpoints and progress tracking.
    """
    
    def __init__(self, endpoint_url: Optional[str] = None, 
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = 'us-west-2'):
        """
        Initialize S3 handler.
        
        Parameters:
        -----------
        endpoint_url : str, optional
            Custom S3 endpoint URL (for S3-compatible services)
        aws_access_key_id : str, optional
            AWS access key ID (will use environment variables if not provided)
        aws_secret_access_key : str, optional
            AWS secret access key (will use environment variables if not provided)
        region_name : str
            AWS region name
        """
        
        self.logger = logging.getLogger('axolotl.s3_handler')
        self.endpoint_url = endpoint_url
        
        # Setup boto3 session with credentials
        session_kwargs = {
            'region_name': region_name
        }
        
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key
            })
        
        try:
            self.session = boto3.Session(**session_kwargs)
            
            # Create S3 client
            client_kwargs = {}
            if endpoint_url:
                client_kwargs['endpoint_url'] = endpoint_url
            
            self.s3_client = self.session.client('s3', **client_kwargs)
            
            # Test connection
            self._test_connection()
            self.logger.info("S3 connection established successfully")
            
        except (NoCredentialsError, ClientError) as e:
            self.logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise
    
    def _test_connection(self):
        """Test S3 connection by listing buckets"""
        try:
            self.s3_client.list_buckets()
        except ClientError as e:
            self.logger.warning(f"S3 connection test failed: {str(e)}")
            # Don't raise error here as some configurations may not allow list_buckets
    
    def _parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """
        Parse S3 path into bucket and key components.
        
        Parameters:
        -----------
        s3_path : str
            S3 path in format s3://bucket/key/path
            
        Returns:
        --------
        tuple
            (bucket_name, key_path)
        """
        
        if not s3_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path: {s3_path}. Must start with 's3://'")
        
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        if not bucket:
            raise ValueError(f"Invalid S3 path: {s3_path}. Bucket name missing")
        
        return bucket, key
    
    def check_exists(self, s3_path: str) -> bool:
        """
        Check if an object exists in S3.
        
        Parameters:
        -----------
        s3_path : str
            S3 path to check
            
        Returns:
        --------
        bool
            True if object exists, False otherwise
        """
        
        try:
            bucket, key = self._parse_s3_path(s3_path)
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                self.logger.error(f"Error checking S3 object existence: {str(e)}")
                raise
        except Exception as e:
            self.logger.error(f"Unexpected error checking S3 object: {str(e)}")
            return False
    
    def upload_file(self, local_path: str, s3_path: str, 
                   progress_callback: Optional[Callable] = None) -> bool:
        """
        Upload a file to S3.
        
        Parameters:
        -----------
        local_path : str
            Local file path to upload
        s3_path : str
            S3 destination path
        progress_callback : callable, optional
            Callback function for progress updates
            
        Returns:
        --------
        bool
            True if upload successful, False otherwise
        """
        
        try:
            bucket, key = self._parse_s3_path(s3_path)
            
            # Get file size for progress tracking
            file_size = os.path.getsize(local_path)
            
            # Setup progress callback
            if progress_callback is None and file_size > 10 * 1024 * 1024:  # 10MB
                # Create default progress bar for large files
                pbar = tqdm(total=file_size, unit='B', unit_scale=True, 
                           desc=f"Uploading {os.path.basename(local_path)}")
                
                def default_callback(bytes_transferred):
                    pbar.update(bytes_transferred)
                
                progress_callback = default_callback
            
            # Upload file
            self.logger.debug(f"Uploading {local_path} to {s3_path}")
            
            # Use direct upload without Callback in ExtraArgs for compatibility
            if progress_callback and file_size:
                # For braingeneers S3, we'll use a simpler approach without progress callback in ExtraArgs
                self.s3_client.upload_file(
                    Filename=local_path,
                    Bucket=bucket,
                    Key=key
                )
                # Update progress bar to completion
                if 'pbar' in locals():
                    pbar.update(file_size)
            else:
                # Simple upload without progress
                self.s3_client.upload_file(
                    Filename=local_path,
                    Bucket=bucket,
                    Key=key
                )
            
            if 'pbar' in locals():
                pbar.close()
            
            self.logger.debug(f"Successfully uploaded {local_path} to {s3_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload {local_path} to {s3_path}: {str(e)}")
            if 'pbar' in locals():
                pbar.close()
            return False
    
    def download_file(self, s3_path: str, local_path: str,
                     progress_callback: Optional[Callable] = None) -> bool:
        """
        Download a file from S3.
        
        Parameters:
        -----------
        s3_path : str
            S3 source path
        local_path : str
            Local destination path
        progress_callback : callable, optional
            Callback function for progress updates
            
        Returns:
        --------
        bool
            True if download successful, False otherwise
        """
        
        try:
            bucket, key = self._parse_s3_path(s3_path)
            
            # Create directory if it doesn't exist
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
            
            # Get file size for progress tracking
            try:
                response = self.s3_client.head_object(Bucket=bucket, Key=key)
                file_size = response['ContentLength']
            except:
                file_size = None
            
            # Setup progress callback
            if progress_callback is None and file_size and file_size > 10 * 1024 * 1024:  # 10MB
                # Create default progress bar for large files
                pbar = tqdm(total=file_size, unit='B', unit_scale=True,
                           desc=f"Downloading {os.path.basename(s3_path)}")
                
                def default_callback(bytes_transferred):
                    pbar.update(bytes_transferred)
                
                progress_callback = default_callback
            
            # Download file
            self.logger.debug(f"Downloading {s3_path} to {local_path}")
            
            # Use direct download without Callback in ExtraArgs for compatibility
            if progress_callback and file_size:
                # For braingeneers S3, we'll use a simpler approach without progress callback in ExtraArgs
                self.s3_client.download_file(
                    Bucket=bucket,
                    Key=key,
                    Filename=local_path
                )
                # Update progress bar to completion
                if 'pbar' in locals():
                    pbar.update(file_size)
            else:
                # Simple download without progress
                self.s3_client.download_file(
                    Bucket=bucket,
                    Key=key,
                    Filename=local_path
                )
            
            if 'pbar' in locals():
                pbar.close()
            
            self.logger.debug(f"Successfully downloaded {s3_path} to {local_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {s3_path} to {local_path}: {str(e)}")
            if 'pbar' in locals():
                pbar.close()
            return False
    
    def upload_directory(self, local_dir: str, s3_prefix: str) -> int:
        """
        Upload a directory to S3.
        
        Parameters:
        -----------
        local_dir : str
            Local directory to upload
        s3_prefix : str
            S3 prefix (bucket and path prefix)
            
        Returns:
        --------
        int
            Number of files successfully uploaded
        """
        
        local_path = Path(local_dir)
        if not local_path.exists() or not local_path.is_dir():
            self.logger.error(f"Local directory does not exist: {local_dir}")
            return 0
        
        uploaded_count = 0
        total_files = sum(1 for _ in local_path.rglob('*') if _.is_file())
        
        self.logger.info(f"Uploading {total_files} files from {local_dir} to {s3_prefix}")
        
        with tqdm(total=total_files, desc="Uploading files") as pbar:
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    # Calculate relative path for S3 key
                    relative_path = file_path.relative_to(local_path)
                    s3_path = f"{s3_prefix.rstrip('/')}/{relative_path.as_posix()}"
                    
                    if self.upload_file(str(file_path), s3_path):
                        uploaded_count += 1
                    
                    pbar.update(1)
        
        self.logger.info(f"Successfully uploaded {uploaded_count}/{total_files} files")
        return uploaded_count
    
    def download_directory(self, s3_prefix: str, local_dir: str) -> int:
        """
        Download a directory from S3.
        
        Parameters:
        -----------
        s3_prefix : str
            S3 prefix to download
        local_dir : str
            Local directory to download to
            
        Returns:
        --------
        int
            Number of files successfully downloaded
        """
        
        try:
            bucket, prefix = self._parse_s3_path(s3_prefix)
            
            # List objects with the prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            all_objects = []
            for page in pages:
                if 'Contents' in page:
                    all_objects.extend(page['Contents'])
            
            if not all_objects:
                self.logger.warning(f"No objects found with prefix: {s3_prefix}")
                return 0
            
            downloaded_count = 0
            self.logger.info(f"Downloading {len(all_objects)} files from {s3_prefix} to {local_dir}")
            
            with tqdm(total=len(all_objects), desc="Downloading files") as pbar:
                for obj in all_objects:
                    key = obj['Key']
                    # Remove prefix to get relative path
                    relative_path = key[len(prefix):].lstrip('/')
                    local_path = os.path.join(local_dir, relative_path)
                    
                    s3_object_path = f"s3://{bucket}/{key}"
                    
                    if self.download_file(s3_object_path, local_path):
                        downloaded_count += 1
                    
                    pbar.update(1)
            
            self.logger.info(f"Successfully downloaded {downloaded_count}/{len(all_objects)} files")
            return downloaded_count
            
        except Exception as e:
            self.logger.error(f"Failed to download directory {s3_prefix}: {str(e)}")
            return 0
    
    def list_objects(self, s3_prefix: str, max_keys: int = 1000) -> list[Dict[str, Any]]:
        """
        List objects in S3 with given prefix.
        
        Parameters:
        -----------
        s3_prefix : str
            S3 prefix to list
        max_keys : int
            Maximum number of keys to return
            
        Returns:
        --------
        list
            List of object metadata dictionaries
        """
        
        try:
            bucket, prefix = self._parse_s3_path(s3_prefix)
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            return response.get('Contents', [])
            
        except Exception as e:
            self.logger.error(f"Failed to list objects with prefix {s3_prefix}: {str(e)}")
            return []
    
    def delete_object(self, s3_path: str) -> bool:
        """
        Delete an object from S3.
        
        Parameters:
        -----------
        s3_path : str
            S3 path to delete
            
        Returns:
        --------
        bool
            True if deletion successful, False otherwise
        """
        
        try:
            bucket, key = self._parse_s3_path(s3_path)
            
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            self.logger.debug(f"Successfully deleted {s3_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete {s3_path}: {str(e)}")
            return False
    
    def get_object_metadata(self, s3_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an S3 object.
        
        Parameters:
        -----------
        s3_path : str
            S3 path to get metadata for
            
        Returns:
        --------
        dict or None
            Object metadata dictionary or None if not found
        """
        
        try:
            bucket, key = self._parse_s3_path(s3_path)
            
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            return response
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            else:
                self.logger.error(f"Error getting metadata for {s3_path}: {str(e)}")
                raise
        except Exception as e:
            self.logger.error(f"Unexpected error getting metadata for {s3_path}: {str(e)}")
            return None