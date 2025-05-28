#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for pre_alignment.py module
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
import sys
import os

# Add parent directory to path to import pre_alignment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess.pre_alignment import pixel_to_coords, utm_to_latlon, process_tiepoints


class TestPixelToCoords:
    """Test pixel_to_coords function"""
    
    def test_pixel_to_coords_basic(self):
        """Test basic pixel to coordinate conversion"""
        # Create a simple affine transform (1m per pixel, origin at 354000, 4107000)
        transform = from_origin(354000, 4107000, 1.0, 1.0)
        
        # Test conversion - xy returns pixel center coordinates
        x, y = pixel_to_coords(0, 0, transform)
        assert x == 354000.5  # pixel center is at 0.5 offset
        assert y == 4106999.5
        
        x, y = pixel_to_coords(100, 200, transform)
        assert x == 354200.5
        assert y == 4106899.5
    
    def test_pixel_to_coords_with_rotation(self):
        """Test pixel to coordinate conversion with more complex transform"""
        # Create affine transform with 0.31m resolution
        transform = from_origin(354234.01, 4108756.10, 0.31, 0.31)
        
        # xy returns pixel center coordinates
        x, y = pixel_to_coords(0, 0, transform)
        assert abs(x - (354234.01 + 0.5 * 0.31)) < 0.001
        assert abs(y - (4108756.10 - 0.5 * 0.31)) < 0.001
        
        x, y = pixel_to_coords(100, 100, transform)
        assert abs(x - (354234.01 + 100.5 * 0.31)) < 0.001
        assert abs(y - (4108756.10 - 100.5 * 0.31)) < 0.001


class TestUtmToLatlon:
    """Test utm_to_latlon function"""
    
    def test_utm_to_latlon_epsg32637(self):
        """Test UTM to lat/lon conversion for EPSG:32637 (WGS 84 / UTM zone 37N)"""
        crs = CRS.from_epsg(32637)
        
        # Test with known coordinates
        # These are approximate values for testing
        lon, lat = utm_to_latlon(354234.01, 4107777.10, crs)
        assert 37.3 < lon < 37.4  # Should be around 37.36°E
        assert 37.0 < lat < 37.2  # Should be around 37.1°N
        
    def test_utm_to_latlon_round_trip(self):
        """Test that conversion is consistent"""
        from pyproj import Transformer
        
        crs = CRS.from_epsg(32637)
        
        # Original UTM coordinates
        utm_x, utm_y = 355000, 4108000
        
        # Convert to lat/lon
        lon, lat = utm_to_latlon(utm_x, utm_y, crs)
        
        # Convert back to UTM
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x_back, y_back = transformer.transform(lon, lat)
        
        # Should be very close to original
        assert abs(x_back - utm_x) < 0.001
        assert abs(y_back - utm_y) < 0.001


class TestProcessTiepoints:
    """Test process_tiepoints function"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory structure with test data"""
        temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        area_dir = Path(temp_dir) / "001"
        area_dir.mkdir(parents=True)
        
        # Create test GeoTIFF files
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'width': 100,
            'height': 100,
            'count': 1,
            'crs': CRS.from_epsg(32637),
            'transform': from_origin(354234.01, 4108756.10, 0.31, 0.31)
        }
        
        # Create optical.tif
        with rasterio.open(area_dir / "0_optical.tif", 'w', **profile) as dst:
            dst.write(np.zeros((100, 100), dtype=np.uint8), 1)
        
        # Create sar.tif with slightly different transform
        profile['transform'] = from_origin(354243.00, 4108715.01, 0.41, 0.41)
        with rasterio.open(area_dir / "1_sar.tif", 'w', **profile) as dst:
            dst.write(np.zeros((100, 100), dtype=np.uint8), 1)
        
        # Create tiepoints CSV
        tiepoints_data = pd.DataFrame({
            'sar_row': [10, 20, 30],
            'sar_col': [15, 25, 35],
            'optical_row': [12, 22, 32],
            'optical_col': [17, 27, 37]
        })
        tiepoints_data.to_csv(area_dir / "2_tiepoints.csv", index=False)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_process_tiepoints_creates_output(self, temp_data_dir):
        """Test that process_tiepoints creates output file"""
        data_dir = Path(temp_data_dir)
        output_dir = data_dir / "output"
        output_dir.mkdir()
        
        # Process tiepoints
        df = process_tiepoints("001", data_dir, output_dir)
        
        # Check output file exists
        output_file = output_dir / "001_tiepoints_converted.csv"
        assert output_file.exists()
        
        # Check dataframe has expected columns
        expected_columns = [
            'sar_row', 'sar_col', 'sar_utm_x', 'sar_utm_y', 'sar_lon', 'sar_lat',
            'optical_row', 'optical_col', 'optical_utm_x', 'optical_utm_y', 'optical_lon', 'optical_lat'
        ]
        assert all(col in df.columns for col in expected_columns)
        
        # Check dataframe has correct number of rows
        assert len(df) == 3
    
    def test_process_tiepoints_coordinates_reasonable(self, temp_data_dir):
        """Test that converted coordinates are in reasonable range"""
        data_dir = Path(temp_data_dir)
        output_dir = data_dir / "output"
        output_dir.mkdir()
        
        # Process tiepoints
        df = process_tiepoints("001", data_dir, output_dir)
        
        # Check UTM coordinates are reasonable
        assert all(354000 < x < 358000 for x in df['sar_utm_x'])
        assert all(354000 < x < 358000 for x in df['optical_utm_x'])
        assert all(4107000 < y < 4109000 for y in df['sar_utm_y'])
        assert all(4107000 < y < 4109000 for y in df['optical_utm_y'])
        
        # Check lat/lon coordinates are reasonable (around 37°N, 37°E)
        assert all(36 < lat < 38 for lat in df['sar_lat'])
        assert all(36 < lat < 38 for lat in df['optical_lat'])
        assert all(36 < lon < 38 for lon in df['sar_lon'])
        assert all(36 < lon < 38 for lon in df['optical_lon'])
    
    def test_process_tiepoints_missing_files(self, temp_data_dir):
        """Test that process_tiepoints raises error for missing files"""
        data_dir = Path(temp_data_dir)
        output_dir = data_dir / "output"
        output_dir.mkdir()
        
        # Try to process non-existent area
        with pytest.raises(FileNotFoundError):
            process_tiepoints("999", data_dir, output_dir)
    
    def test_process_tiepoints_preserves_pixel_coords(self, temp_data_dir):
        """Test that original pixel coordinates are preserved in output"""
        data_dir = Path(temp_data_dir)
        output_dir = data_dir / "output"
        output_dir.mkdir()
        
        # Process tiepoints
        df = process_tiepoints("001", data_dir, output_dir)
        
        # Original tiepoints
        original_df = pd.read_csv(data_dir / "001" / "2_tiepoints.csv")
        
        # Check that pixel coordinates are preserved
        assert list(df['sar_row']) == list(original_df['sar_row'])
        assert list(df['sar_col']) == list(original_df['sar_col'])
        assert list(df['optical_row']) == list(original_df['optical_row'])
        assert list(df['optical_col']) == list(original_df['optical_col'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])