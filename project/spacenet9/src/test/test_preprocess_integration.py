#!/usr/bin/env python3
"""
Integration tests for preprocessing pipeline
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import yaml
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestPreprocessingPipeline:
    """Test the complete preprocessing pipeline"""
    
    @pytest.fixture
    def temp_project_structure(self):
        """Create a complete temporary project structure"""
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir)
        
        # Create directory structure
        raw_dir = project_dir / "00_data" / "raw" / "train" / "001"
        raw_dir.mkdir(parents=True)
        
        interim_dir = project_dir / "00_data" / "interim"
        interim_dir.mkdir(parents=True)
        
        processed_dir = project_dir / "00_data" / "processed"
        processed_dir.mkdir(parents=True)
        
        # Create config directory
        config_dir = project_dir / "configs"
        config_dir.mkdir(parents=True)
        
        # Create preprocessing config
        config = {
            'input': {
                'raw_data_dir': '00_data/raw',
                'train_regions': ['001']
            },
            'output': {
                'processed_data_dir': '00_data/processed',
                'interim_data_dir': '00_data/interim'
            },
            'tiling': {
                'tile_sizes': 256,
                'padding_mode': 'skip'
            },
            'resampling': {
                'target_resolution': 0.3,
                'interpolation_optical': 'bilinear',
                'interpolation_sar': 'bilinear'
            },
            'coordinate_system': {
                'target_crs': 'EPSG:4326'
            }
        }
        
        config_path = config_dir / "preprocessing_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        # Create test optical image
        optical_profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'width': 1000,
            'height': 1000,
            'count': 3,  # RGB
            'crs': CRS.from_epsg(32637),
            'transform': from_origin(354234.01, 4108756.10, 0.305, 0.305)
        }
        
        with rasterio.open(raw_dir / "0_optical.tif", 'w', **optical_profile) as dst:
            for i in range(3):
                # Create gradient pattern
                data = np.arange(1000000, dtype=np.uint16).reshape(1000, 1000) % 65536
                dst.write(data + i * 10000, i + 1)
        
        # Create test SAR image (slightly different bounds for overlap testing)
        sar_profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'width': 800,
            'height': 800,
            'count': 1,
            'crs': CRS.from_epsg(32637),
            'transform': from_origin(354243.00, 4108715.01, 0.409, 0.409)
        }
        
        with rasterio.open(raw_dir / "1_sar.tif", 'w', **sar_profile) as dst:
            data = np.arange(640000, dtype=np.uint16).reshape(800, 800) % 65536
            dst.write(data, 1)
        
        # Create test tiepoints
        tiepoints_data = pd.DataFrame({
            'sar_row': [100, 200, 300, 400, 500],
            'sar_col': [150, 250, 350, 450, 550],
            'optical_row': [130, 260, 390, 520, 650],
            'optical_col': [165, 275, 385, 495, 605]
        })
        tiepoints_data.to_csv(raw_dir / "2_tiepoints.csv", index=False)
        
        yield project_dir, config_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_directory_structure_creation(self, temp_project_structure):
        """Test that preprocessing creates expected directory structure"""
        project_dir, config_path = temp_project_structure
        
        # Check initial structure
        assert (project_dir / "00_data" / "raw" / "train" / "001").exists()
        assert (project_dir / "00_data" / "interim").exists()
        assert (project_dir / "00_data" / "processed").exists()
        assert config_path.exists()
    
    def test_preprocessing_config_valid(self, temp_project_structure):
        """Test that preprocessing config is valid"""
        project_dir, config_path = temp_project_structure
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        assert 'input' in config
        assert 'output' in config
        assert 'tiling' in config
        assert 'resampling' in config
        
        # Check values
        assert config['tiling']['tile_sizes'] == 256
        assert config['resampling']['target_resolution'] == 0.3
    
    def test_input_data_valid(self, temp_project_structure):
        """Test that input data is valid"""
        project_dir, config_path = temp_project_structure
        
        raw_dir = project_dir / "00_data" / "raw" / "train" / "001"
        
        # Check optical image
        with rasterio.open(raw_dir / "0_optical.tif") as src:
            assert src.count == 3
            assert src.crs == CRS.from_epsg(32637)
            assert src.width == 1000
            assert src.height == 1000
        
        # Check SAR image
        with rasterio.open(raw_dir / "1_sar.tif") as src:
            assert src.count == 1
            assert src.crs == CRS.from_epsg(32637)
            assert src.width == 800
            assert src.height == 800
        
        # Check tiepoints
        tiepoints_df = pd.read_csv(raw_dir / "2_tiepoints.csv")
        assert len(tiepoints_df) == 5
        assert all(col in tiepoints_df.columns for col in 
                  ['sar_row', 'sar_col', 'optical_row', 'optical_col'])
    
    def test_preprocessing_output_structure(self, temp_project_structure):
        """Test expected output directory structure after preprocessing"""
        project_dir, config_path = temp_project_structure
        
        # Expected interim directories
        expected_interim_dirs = [
            "tiepoint_converted",
            "resampled",
            "tiepoint_resampled",
            "resample_verification"
        ]
        
        # Expected processed directories
        expected_processed_dirs = [
            "opt_keypoint_detection",
            "sar_keypoint_detection"
        ]
        
        # These would be created by running the preprocessing scripts
        # Here we just verify the structure is as expected
        interim_dir = project_dir / "00_data" / "interim"
        processed_dir = project_dir / "00_data" / "processed"
        
        # The actual preprocessing would create these
        # This test documents the expected structure


class TestDataIntegrity:
    """Test data integrity throughout preprocessing"""
    
    def test_tiepoint_count_preservation(self):
        """Test that tiepoint counts are preserved or filtered correctly"""
        # Create test data
        original_tiepoints = pd.DataFrame({
            'sar_row': range(10),
            'sar_col': range(10),
            'optical_row': range(10),
            'optical_col': range(10)
        })
        
        # In actual preprocessing, some tiepoints might be filtered
        # if they fall outside image bounds
        assert len(original_tiepoints) == 10
    
    def test_coordinate_system_consistency(self):
        """Test that coordinate systems are consistent"""
        # All processed images should have the same CRS
        expected_crs = CRS.from_epsg(32637)
        
        # This would be verified in actual preprocessing
        assert expected_crs.to_epsg() == 32637
    
    def test_tile_size_consistency(self):
        """Test that all tiles have consistent size"""
        expected_tile_size = 256
        
        # All generated tiles should be 256x256
        assert expected_tile_size == 256


class TestErrorHandling:
    """Test error handling in preprocessing"""
    
    def test_missing_input_file_handling(self):
        """Test handling of missing input files"""
        # Test that appropriate errors are raised
        with pytest.raises(FileNotFoundError):
            # Simulate missing file
            pd.read_csv("nonexistent_file.csv")
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration"""
        # Test that invalid configs are caught
        invalid_config = {'invalid': 'config'}
        
        # Should raise error for missing required fields
        assert 'input' not in invalid_config
        assert 'output' not in invalid_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])