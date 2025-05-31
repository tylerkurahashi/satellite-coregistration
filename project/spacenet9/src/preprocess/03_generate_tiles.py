#!/usr/bin/env python3
"""
SpaceNet9 タイル生成とキーポイント検出データセット作成（地理情報付き）

リサンプル済みの光学画像とSAR画像から、タイポイントを中心とした
タイルを生成し、キーポイント検出用のデータセットを作成する。
各タイルには地理情報（CRS、変換行列）を付与する。
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import yaml
import argparse
import logging
from typing import Tuple, Dict, Optional
import json
from scipy.stats import multivariate_normal


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


def generate_gaussian_heatmap(
    center_x: float,
    center_y: float,
    size: int,
    sigma: float = 13.0
) -> np.ndarray:
    """
    ガウシアンヒートマップを生成する
    
    Args:
        center_x: 中心X座標
        center_y: 中心Y座標
        size: 画像サイズ
        sigma: 標準偏差（分散はsigma^2）
    
    Returns:
        ガウシアンヒートマップ
    """
    # メッシュグリッドを作成
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    pos = np.dstack((x, y))
    
    # 2次元ガウシアン分布を計算
    rv = multivariate_normal([center_x, center_y], [[sigma**2, 0], [0, sigma**2]])
    heatmap = rv.pdf(pos)
    
    # 最大値で正規化
    heatmap = heatmap / np.max(heatmap)
    
    return heatmap.astype(np.float32)


def extract_tile_with_geo(
    image_path: Path,
    center_row: int,
    center_col: int,
    tile_size: int = 256
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    画像から指定位置を中心としたタイルを抽出し、地理情報も取得する
    
    Args:
        image_path: 画像パス
        center_row: 中心行
        center_col: 中心列
        tile_size: タイルサイズ
    
    Returns:
        抽出されたタイルとメタデータ（境界外の場合はNone）
    """
    half_size = tile_size // 2
    
    with rasterio.open(image_path) as src:
        # タイルの範囲を計算
        row_start = center_row - half_size
        col_start = center_col - half_size
        
        # 境界チェック
        if (row_start < 0 or col_start < 0 or 
            row_start + tile_size > src.height or 
            col_start + tile_size > src.width):
            return None, None
        
        # ウィンドウを定義
        window = Window(col_start, row_start, tile_size, tile_size)
        
        # タイルの変換行列を計算
        tile_transform = rasterio.windows.transform(window, src.transform)
        
        # タイルのメタデータを作成
        tile_meta = src.meta.copy()
        tile_meta.update({
            'height': tile_size,
            'width': tile_size,
            'transform': tile_transform
        })
        
        # データを読み込む
        if src.count >= 3:
            # RGB画像の場合
            data = np.stack([src.read(i, window=window) for i in [1, 2, 3]], axis=-1)
        else:
            # グレースケール画像の場合
            data = src.read(1, window=window)
            
    return data, tile_meta


def save_tile_with_geo(
    tile_data: np.ndarray,
    output_path: Path,
    meta: Dict
) -> None:
    """地理情報付きでタイルを保存する"""
    os.makedirs(output_path.parent, exist_ok=True)
    
    # メタデータを更新
    profile = meta.copy()
    profile.update({
        'driver': 'GTiff',
        'compress': 'lzw'
    })
    
    if len(tile_data.shape) == 3:
        # RGB画像
        profile.update({
            'count': tile_data.shape[2],
            'dtype': tile_data.dtype
        })
    else:
        # グレースケール画像
        profile.update({
            'count': 1,
            'dtype': tile_data.dtype
        })
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        if len(tile_data.shape) == 3:
            # RGB画像
            for i in range(tile_data.shape[2]):
                dst.write(tile_data[:, :, i], i + 1)
        else:
            # グレースケール画像
            dst.write(tile_data, 1)


def generate_optical_keypoint_dataset(
    region: str,
    tiepoints_df: pd.DataFrame,
    optical_path: Path,
    sar_path: Path,
    output_dir: Path,
    tile_size: int = 256,
    sigma: float = 13.0
) -> Dict:
    """
    光学キーポイント検出データセットを生成する（地理情報付き）
    
    SAR画像の各Tiepointを中心にタイルを生成し、
    対応する光学画像のTiepointを中心としたガウシアンヒートマップを作成
    """
    dataset_info = {
        'region': region,
        'tile_size': tile_size,
        'sigma': sigma,
        'tiles': []
    }
    
    region_output_dir = output_dir / region
    os.makedirs(region_output_dir, exist_ok=True)
    
    # 元画像のCRSを取得
    with rasterio.open(optical_path) as src:
        crs = src.crs
        dataset_info['crs'] = str(crs)
    
    for idx, row in tiepoints_df.iterrows():
        # SAR画像のタイポイントを中心にタイルを抽出
        sar_tile, sar_meta = extract_tile_with_geo(
            sar_path, int(row['sar_row']), int(row['sar_col']), tile_size
        )
        optical_tile, optical_meta = extract_tile_with_geo(
            optical_path, int(row['sar_row']), int(row['sar_col']), tile_size
        )
        
        if sar_tile is None or optical_tile is None:
            logging.warning(f"タイル {idx} をスキップ: 境界外")
            continue
        
        # 光学画像のタイポイント位置を計算（SAR座標系での相対位置）
        relative_opt_row = int(row['optical_row']) - int(row['sar_row']) + tile_size // 2
        relative_opt_col = int(row['optical_col']) - int(row['sar_col']) + tile_size // 2
        
        # ガウシアンヒートマップを生成
        heatmap = generate_gaussian_heatmap(
            relative_opt_col, relative_opt_row, tile_size, sigma
        )
        
        # ヒートマップ用のメタデータを作成（座標系なし）
        heatmap_meta = sar_meta.copy()
        heatmap_meta.update({
            'count': 1,
            'dtype': 'float32'
        })
        
        # タイルを保存
        tile_name = f'tile_{idx:04d}'
        sar_tile_path = region_output_dir / f'{tile_name}_sar.tif'
        optical_tile_path = region_output_dir / f'{tile_name}_optical.tif'
        heatmap_path = region_output_dir / f'{tile_name}_heatmap.tif'
        
        save_tile_with_geo(sar_tile, sar_tile_path, sar_meta)
        save_tile_with_geo(optical_tile, optical_tile_path, optical_meta)
        save_tile_with_geo(heatmap, heatmap_path, heatmap_meta)
        
        # タイルの地理範囲を計算
        bounds = rasterio.transform.array_bounds(
            tile_size, tile_size, sar_meta['transform']
        )
        
        # データセット情報を更新
        tile_info = {
            'index': idx,
            'sar_tile': str(sar_tile_path.name),
            'optical_tile': str(optical_tile_path.name),
            'heatmap': str(heatmap_path.name),
            'sar_center': {'row': int(row['sar_row']), 'col': int(row['sar_col'])},
            'optical_center': {'row': int(row['optical_row']), 'col': int(row['optical_col'])},
            'relative_optical_center': {'row': relative_opt_row, 'col': relative_opt_col},
            'bounds': {
                'left': bounds[0],
                'bottom': bounds[1],
                'right': bounds[2],
                'top': bounds[3]
            },
            'transform': list(sar_meta['transform'])
        }
        dataset_info['tiles'].append(tile_info)
    
    # データセット情報を保存
    info_path = region_output_dir / 'dataset_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2)
    
    logging.info(f"光学キーポイント検出データセット生成完了: {len(dataset_info['tiles'])}タイル")
    
    return dataset_info


def generate_sar_keypoint_dataset(
    region: str,
    tiepoints_df: pd.DataFrame,
    optical_path: Path,
    sar_path: Path,
    output_dir: Path,
    tile_size: int = 256,
    sigma: float = 13.0
) -> Dict:
    """
    SARキーポイント検出データセットを生成する（地理情報付き）
    
    光学画像の各Tiepointを中心にタイルを生成し、
    対応するSAR画像のTiepointを中心としたガウシアンヒートマップを作成
    """
    dataset_info = {
        'region': region,
        'tile_size': tile_size,
        'sigma': sigma,
        'tiles': []
    }
    
    region_output_dir = output_dir / region
    os.makedirs(region_output_dir, exist_ok=True)
    
    # 元画像のCRSを取得
    with rasterio.open(optical_path) as src:
        crs = src.crs
        dataset_info['crs'] = str(crs)
    
    for idx, row in tiepoints_df.iterrows():
        # 光学画像のタイポイントを中心にタイルを抽出
        optical_tile, optical_meta = extract_tile_with_geo(
            optical_path, int(row['optical_row']), int(row['optical_col']), tile_size
        )
        sar_tile, sar_meta = extract_tile_with_geo(
            sar_path, int(row['optical_row']), int(row['optical_col']), tile_size
        )
        
        if optical_tile is None or sar_tile is None:
            logging.warning(f"タイル {idx} をスキップ: 境界外")
            continue
        
        # SAR画像のタイポイント位置を計算（光学座標系での相対位置）
        relative_sar_row = int(row['sar_row']) - int(row['optical_row']) + tile_size // 2
        relative_sar_col = int(row['sar_col']) - int(row['optical_col']) + tile_size // 2
        
        # ガウシアンヒートマップを生成
        heatmap = generate_gaussian_heatmap(
            relative_sar_col, relative_sar_row, tile_size, sigma
        )
        
        # ヒートマップ用のメタデータを作成
        heatmap_meta = optical_meta.copy()
        heatmap_meta.update({
            'count': 1,
            'dtype': 'float32'
        })
        
        # タイルを保存
        tile_name = f'tile_{idx:04d}'
        optical_tile_path = region_output_dir / f'{tile_name}_optical.tif'
        sar_tile_path = region_output_dir / f'{tile_name}_sar.tif'
        heatmap_path = region_output_dir / f'{tile_name}_heatmap.tif'
        
        save_tile_with_geo(optical_tile, optical_tile_path, optical_meta)
        save_tile_with_geo(sar_tile, sar_tile_path, sar_meta)
        save_tile_with_geo(heatmap, heatmap_path, heatmap_meta)
        
        # タイルの地理範囲を計算
        bounds = rasterio.transform.array_bounds(
            tile_size, tile_size, optical_meta['transform']
        )
        
        # データセット情報を更新
        tile_info = {
            'index': idx,
            'optical_tile': str(optical_tile_path.name),
            'sar_tile': str(sar_tile_path.name),
            'heatmap': str(heatmap_path.name),
            'optical_center': {'row': int(row['optical_row']), 'col': int(row['optical_col'])},
            'sar_center': {'row': int(row['sar_row']), 'col': int(row['sar_col'])},
            'relative_sar_center': {'row': relative_sar_row, 'col': relative_sar_col},
            'bounds': {
                'left': bounds[0],
                'bottom': bounds[1],
                'right': bounds[2],
                'top': bounds[3]
            },
            'transform': list(optical_meta['transform'])
        }
        dataset_info['tiles'].append(tile_info)
    
    # データセット情報を保存
    info_path = region_output_dir / 'dataset_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2)
    
    logging.info(f"SARキーポイント検出データセット生成完了: {len(dataset_info['tiles'])}タイル")
    
    return dataset_info


def process_region(region: str, config: Dict, base_dir: Path) -> None:
    """1つの地域のデータを処理する"""
    
    logging.info(f"地域 {region} の処理を開始します")
    
    # パスの設定
    interim_dir = base_dir / config['output']['interim_data_dir']
    processed_dir = base_dir / config['output']['processed_data_dir']
    
    resampled_dir = interim_dir / 'resampled' / region
    tiepoint_resampled_dir = interim_dir / 'tiepoint_resampled'
    
    optical_path = resampled_dir / '0_optical_resampled.tif'
    sar_path = resampled_dir / '1_sar_resampled.tif'
    tiepoints_path = tiepoint_resampled_dir / f'{region}_tiepoints_resampled.csv'
    
    # データを読み込む
    tiepoints_df = pd.read_csv(tiepoints_path)
    
    # 光学キーポイント検出データセットを生成
    opt_output_dir = processed_dir / 'opt_keypoint_detection'
    generate_optical_keypoint_dataset(
        region, tiepoints_df, optical_path, sar_path,
        opt_output_dir, config['tiling']['tile_sizes']
    )
    
    # SARキーポイント検出データセットを生成
    sar_output_dir = processed_dir / 'sar_keypoint_detection'
    generate_sar_keypoint_dataset(
        region, tiepoints_df, optical_path, sar_path,
        sar_output_dir, config['tiling']['tile_sizes']
    )


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='SpaceNet9 タイル生成（地理情報付き）')
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