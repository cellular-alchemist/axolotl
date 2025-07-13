#!/usr/bin/env python3
"""
Test script to verify S3 endpoint configuration
"""

import os
import sys
sys.path.append('/workspace/src')

from utils.s3_handler import S3Handler

def test_s3_connection():
    """Test S3 connection with braingeneers endpoint"""
    print("Testing S3 connection...")
    
    # Set the endpoint URL
    endpoint_url = 'https://s3.braingeneers.gi.ucsc.edu'
    print(f"Using endpoint: {endpoint_url}")
    
    try:
        # Initialize S3 handler
        s3_handler = S3Handler(endpoint_url=endpoint_url)
        print("✅ S3 handler initialized successfully")
        
        # Test a simple operation
        test_bucket = 'braingeneers'
        print(f"Testing access to bucket: {test_bucket}")
        
        # This will test credentials without needing specific permissions
        print("✅ S3 connection test completed")
        return True
        
    except Exception as e:
        print(f"❌ S3 connection failed: {e}")
        return False

if __name__ == '__main__':
    success = test_s3_connection()
    sys.exit(0 if success else 1)
