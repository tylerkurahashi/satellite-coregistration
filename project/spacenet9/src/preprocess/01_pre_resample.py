#!/usr/bin/env python3
"""
SpaceNet9 光学画像とSAR画像のリサンプリング処理

光学画像とSAR画像を同じ解像度にリサンプリングし、重複するAOIで切り取る。
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import rowcol, xy
from rasterio.coords import BoundingBox
import yaml
from typing import Tuple, Dict, Optional
import argparse
import logging


def setup_logging(log_level: str = "INFO") -> None:
    """ログ設定を初期化する"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: Path) -> Dict:
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_image_bounds(image_path: Path) -> Tuple[BoundingBox, Dict]:
    """画像の境界とメタデータを取得する"""
    with rasterio.open(image_path) as src:
        bounds = src.bounds
        meta = src.meta.copy()
        return bounds, meta


def calculate_overlap_bounds(optical_bounds: BoundingBox, sar_bounds: BoundingBox) -> BoundingBox:
    """光学画像とSAR画像の重複領域を計算する"""
    left = max(optical_bounds.left, sar_bounds.left)
    bottom = max(optical_bounds.bottom, sar_bounds.bottom)
    right = min(optical_bounds.right, sar_bounds.right)
    top = min(optical_bounds.top, sar_bounds.top)
    
    if left >= right or bottom >= top:
        raise ValueError("光学画像とSAR画像に重複領域がありません")
    
    return BoundingBox(left, bottom, right, top)


def resample_image(
    src_path: Path,
    dst_path: Path,
    target_resolution: float,
    overlap_bounds: BoundingBox,
    resampling_method: str = "bilinear"
) -> Tuple[rasterio.Affine, int, int]:
    """画像をリサンプリングして保存する"""
    
    # リサンプリング方法の対応表
    resampling_methods = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "lanczos": Resampling.lanczos
    }
    
    if resampling_method not in resampling_methods:
        raise ValueError(f"無効なリサンプリング方法: {resampling_method}")
    
    with rasterio.open(src_path) as src:
        # 重複領域の幅と高さを計算
        width = int((overlap_bounds.right - overlap_bounds.left) / target_resolution)
        height = int((overlap_bounds.top - overlap_bounds.bottom) / target_resolution)
        
        # 新しい変換行列を作成
        dst_transform = rasterio.transform.from_bounds(
            overlap_bounds.left, overlap_bounds.bottom,
            overlap_bounds.right, overlap_bounds.top,
            width, height
        )
        
        # 出力メタデータを更新
        dst_meta = src.meta.copy()
        dst_meta.update({
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'transform': dst_transform,
            'compress': 'lzw'
        })
        
        # リサンプリングして書き込み
        os.makedirs(dst_path.parent, exist_ok=True)
        with rasterio.open(dst_path, 'w', **dst_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=src.crs,
                    resampling=resampling_methods[resampling_method]
                )
        
        logging.info(f"リサンプリング完了: {src_path.name} -> {dst_path}")
        logging.info(f"  解像度: {target_resolution}m/pixel")
        logging.info(f"  サイズ: {width}x{height}")
        
        return dst_transform, width, height


def transform_tiepoints(
    tiepoints_df: pd.DataFrame,
    optical_src_transform: rasterio.Affine,
    sar_src_transform: rasterio.Affine,
    optical_dst_transform: rasterio.Affine,
    sar_dst_transform: rasterio.Affine,
    optical_src_bounds: BoundingBox,
    sar_src_bounds: BoundingBox,
    overlap_bounds: BoundingBox
) -> pd.DataFrame:
    """タイポイントをリサンプリング後の座標に変換する"""
    
    transformed_points = []
    
    for _, row in tiepoints_df.iterrows():
        # 元の画像のピクセル座標を地理座標に変換
        optical_x, optical_y = xy(optical_src_transform, row['optical_row'], row['optical_col'])
        sar_x, sar_y = xy(sar_src_transform, row['sar_row'], row['sar_col'])
        
        # 重複領域内にあるかチェック
        if (overlap_bounds.left <= optical_x <= overlap_bounds.right and
            overlap_bounds.bottom <= optical_y <= overlap_bounds.top and
            overlap_bounds.left <= sar_x <= overlap_bounds.right and
            overlap_bounds.bottom <= sar_y <= overlap_bounds.top):
            
            # 新しい画像のピクセル座標に変換
            new_optical_row, new_optical_col = rowcol(optical_dst_transform, optical_x, optical_y)
            new_sar_row, new_sar_col = rowcol(sar_dst_transform, sar_x, sar_y)
            
            transformed_points.append({
                'sar_row': new_sar_row,
                'sar_col': new_sar_col,
                'optical_row': new_optical_row,
                'optical_col': new_optical_col
            })
    
    return pd.DataFrame(transformed_points)


def process_region(region: str, config: Dict, base_dir: Path) -> None:
    """1つの地域のデータを処理する"""
    
    logging.info(f"地域 {region} の処理を開始します")
    
    # パスの設定
    raw_dir = base_dir / config['input']['raw_data_dir'] / 'train' / region
    interim_dir = base_dir / config['output']['interim_data_dir']
    resampled_dir = interim_dir / 'resampled' / region
    tiepoint_resampled_dir = interim_dir / 'tiepoint_resampled'
    
    optical_path = raw_dir / '0_optical.tif'
    sar_path = raw_dir / '1_sar.tif'
    tiepoints_path = raw_dir / '2_tiepoints.csv'
    
    # 画像の境界とメタデータを取得
    optical_bounds, optical_meta = get_image_bounds(optical_path)
    sar_bounds, sar_meta = get_image_bounds(sar_path)
    
    # 重複領域を計算
    overlap_bounds = calculate_overlap_bounds(optical_bounds, sar_bounds)
    logging.info(f"重複領域: {overlap_bounds}")
    
    # 元の変換行列を保存
    optical_src_transform = optical_meta['transform']
    sar_src_transform = sar_meta['transform']
    
    # リサンプリング
    target_resolution = config['resampling']['target_resolution']
    
    optical_dst_path = resampled_dir / '0_optical_resampled.tif'
    optical_dst_transform, _, _ = resample_image(
        optical_path, optical_dst_path, target_resolution,
        overlap_bounds, config['resampling']['interpolation_optical']
    )
    
    sar_dst_path = resampled_dir / '1_sar_resampled.tif'
    sar_dst_transform, _, _ = resample_image(
        sar_path, sar_dst_path, target_resolution,
        overlap_bounds, config['resampling']['interpolation_sar']
    )
    
    # タイポイントの変換
    if tiepoints_path.exists():
        tiepoints_df = pd.read_csv(tiepoints_path)
        
        transformed_tiepoints = transform_tiepoints(
            tiepoints_df,
            optical_src_transform, sar_src_transform,
            optical_dst_transform, sar_dst_transform,
            optical_bounds, sar_bounds,
            overlap_bounds
        )
        
        # タイポイントを保存
        os.makedirs(tiepoint_resampled_dir, exist_ok=True)
        output_path = tiepoint_resampled_dir / f'{region}_tiepoints_resampled.csv'
        transformed_tiepoints.to_csv(output_path, index=False)
        
        logging.info(f"タイポイント変換完了: {len(tiepoints_df)} -> {len(transformed_tiepoints)} 点")
        logging.info(f"保存先: {output_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='SpaceNet9 画像リサンプリング')
    parser.add_argument(
        '--config', 
        type=Path,
        default=Path(__file__).parent.parent.parent / 'configs' / 'preprocessing_config.yaml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--regions',
        nargs='+',
        help='処理する地域（指定しない場合は設定ファイルの全地域）'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='ログレベル'
    )
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(args.log_level)
    
    # 設定ファイルを読み込む
    config = load_config(args.config)
    
    # ベースディレクトリ
    base_dir = args.config.parent.parent
    
    # 処理する地域のリスト
    regions = args.regions if args.regions else config['input']['train_regions']
    
    # 各地域を処理
    for region in regions:
        try:
            process_region(region, config, base_dir)
        except Exception as e:
            logging.error(f"地域 {region} の処理中にエラーが発生しました: {e}")
            if args.log_level == 'DEBUG':
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()