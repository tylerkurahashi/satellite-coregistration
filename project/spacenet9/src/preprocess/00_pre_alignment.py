#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SpaceNet9 Tiepoint coordinate transformation

Convert tiepoint pixel coordinates to latitude/longitude coordinates
"""

import os
import json
import pandas as pd
import rasterio
from rasterio.transform import xy
from pathlib import Path
import numpy as np
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def pixel_to_coords(row: int, col: int, transform) -> Tuple[float, float]:
    """
    Convert pixel coordinates to geographic coordinates
    
    Args:
        row: Image row number (y coordinate)
        col: Image column number (x coordinate)
        transform: rasterio affine transformation matrix
        
    Returns:
        (x, y): Geographic coordinates (UTM coordinate system)
    """
    x, y = xy(transform, row, col)
    return x, y


def utm_to_latlon(x: float, y: float, crs) -> Tuple[float, float]:
    """
    Convert UTM coordinates to latitude/longitude
    
    Args:
        x: UTM X coordinate
        y: UTM Y coordinate
        crs: Coordinate reference system
        
    Returns:
        (lon, lat): Longitude, latitude
    """
    from pyproj import Transformer
    
    # Create transformer from UTM to WGS84
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    
    return lon, lat


def create_wkt_point(lon: float, lat: float) -> str:
    """
    Create WKT POINT string from longitude and latitude
    
    Args:
        lon: Longitude
        lat: Latitude
        
    Returns:
        WKT POINT string
    """
    return f"POINT ({lon} {lat})"


def create_geojson_feature(lon: float, lat: float, properties: Dict) -> Dict:
    """
    Create GeoJSON feature from coordinates and properties
    
    Args:
        lon: Longitude
        lat: Latitude
        properties: Feature properties
        
    Returns:
        GeoJSON feature dictionary
    """
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        },
        "properties": properties
    }


def save_geojson(features: List[Dict], output_path: Path):
    """
    Save features as GeoJSON file
    
    Args:
        features: List of GeoJSON features
        output_path: Path to save GeoJSON file
    """
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)


def process_tiepoints(area_id: str, data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """
    Process tiepoints for a specific area and perform coordinate transformation
    
    Args:
        area_id: Area ID (e.g., "001")
        data_dir: Raw data directory
        output_dir: Output directory
        
    Returns:
        Transformed tiepoint dataframe
    """
    # Set file paths
    area_path = data_dir / area_id
    optical_path = area_path / "0_optical.tif"
    sar_path = area_path / "1_sar.tif"
    tiepoint_path = area_path / "2_tiepoints.csv"
    
    # Check file existence
    if not all(p.exists() for p in [optical_path, sar_path, tiepoint_path]):
        raise FileNotFoundError(f"Required files not found: {area_id}")
    
    logger.info(f"Starting processing for area {area_id}")
    
    # Load tiepoint data
    tiepoints_df = pd.read_csv(tiepoint_path)
    logger.info(f"Number of tiepoints: {len(tiepoints_df)}")
    
    # Get GeoTIFF metadata
    with rasterio.open(optical_path) as optical_src:
        optical_transform = optical_src.transform
        optical_crs = optical_src.crs
        
    with rasterio.open(sar_path) as sar_src:
        sar_transform = sar_src.transform
        sar_crs = sar_src.crs
    
    # Perform coordinate transformation
    results = []
    sar_features = []
    optical_features = []
    
    for idx, row in tiepoints_df.iterrows():
        # Convert SAR image pixel coordinates to UTM coordinates
        sar_x, sar_y = pixel_to_coords(row['sar_row'], row['sar_col'], sar_transform)
        # Convert UTM coordinates to latitude/longitude
        sar_lon, sar_lat = utm_to_latlon(sar_x, sar_y, sar_crs)
        
        # Convert Optical image pixel coordinates to UTM coordinates
        optical_x, optical_y = pixel_to_coords(row['optical_row'], row['optical_col'], optical_transform)
        # Convert UTM coordinates to latitude/longitude
        optical_lon, optical_lat = utm_to_latlon(optical_x, optical_y, optical_crs)
        
        # Create WKT POINT strings
        sar_wkt = create_wkt_point(sar_lon, sar_lat)
        optical_wkt = create_wkt_point(optical_lon, optical_lat)
        
        results.append({
            'sar_row': row['sar_row'],
            'sar_col': row['sar_col'],
            'sar_utm_x': sar_x,
            'sar_utm_y': sar_y,
            'sar_lon': sar_lon,
            'sar_lat': sar_lat,
            'sar_wkt': sar_wkt,
            'optical_row': row['optical_row'],
            'optical_col': row['optical_col'],
            'optical_utm_x': optical_x,
            'optical_utm_y': optical_y,
            'optical_lon': optical_lon,
            'optical_lat': optical_lat,
            'optical_wkt': optical_wkt
        })
        
        # Create GeoJSON features
        sar_feature = create_geojson_feature(sar_lon, sar_lat, {
            'tiepoint_id': int(idx),
            'row': int(row['sar_row']),
            'col': int(row['sar_col']),
            'utm_x': float(sar_x),
            'utm_y': float(sar_y),
            'source': 'sar'
        })
        sar_features.append(sar_feature)
        
        optical_feature = create_geojson_feature(optical_lon, optical_lat, {
            'tiepoint_id': int(idx),
            'row': int(row['optical_row']),
            'col': int(row['optical_col']),
            'utm_x': float(optical_x),
            'utm_y': float(optical_y),
            'source': 'optical'
        })
        optical_features.append(optical_feature)
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    
    # Save output files
    # CSV file with all data including WKT
    output_path = output_dir / f"{area_id}_tiepoints_converted.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved conversion results: {output_path}")
    
    # GeoJSON files
    sar_geojson_path = output_dir / f"{area_id}_tiepoints_sar.geojson"
    save_geojson(sar_features, sar_geojson_path)
    logger.info(f"Saved SAR GeoJSON: {sar_geojson_path}")
    
    optical_geojson_path = output_dir / f"{area_id}_tiepoints_optical.geojson"
    save_geojson(optical_features, optical_geojson_path)
    logger.info(f"Saved Optical GeoJSON: {optical_geojson_path}")
    
    return results_df


def main():
    """Main processing"""
    # Set paths
    base_dir = Path(__file__).parent.parent.parent / "00_data"
    raw_dir = base_dir / "raw" / "train"
    output_dir = base_dir / "interim" / "tiepoint_converted"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each area
    areas = ["001", "002", "003"]
    
    for area_id in areas:
        try:
            df = process_tiepoints(area_id, raw_dir, output_dir)
            logger.info(f"Completed processing for area {area_id}: {len(df)} tiepoints")
        except Exception as e:
            logger.error(f"Error processing area {area_id}: {e}")
            continue
    
    logger.info("All areas processing completed")


if __name__ == "__main__":
    main()
