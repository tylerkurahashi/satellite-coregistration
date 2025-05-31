#!/usr/bin/env python3
"""
SpaceNet9 リサンプリング結果の確認

リサンプル後の光学画像とSAR画像にタイポイントを重ね合わせて表示し、
対応が正しいことを確認する。
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, ConnectionPatch
import yaml
import argparse
import logging
from typing import Tuple, List, Dict


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


def load_resampled_image(image_path: Path) -> Tuple[np.ndarray, rasterio.Affine]:
    """リサンプル済み画像を読み込む"""
    with rasterio.open(image_path) as src:
        # RGB画像の場合は3バンド、SAR画像の場合は1バンドを読み込む
        if src.count >= 3:
            # RGBの場合
            data = np.stack([src.read(i) for i in [1, 2, 3]], axis=-1)
        else:
            # グレースケールの場合
            data = src.read(1)
        transform = src.transform
        
    return data, transform


def normalize_image(image: np.ndarray, percentile_low: float = 2, percentile_high: float = 98) -> np.ndarray:
    """画像を正規化する"""
    # パーセンタイルでクリップ
    if len(image.shape) == 3:
        # RGB画像の場合
        normalized = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            channel = image[:, :, i]
            p_low = np.percentile(channel, percentile_low)
            p_high = np.percentile(channel, percentile_high)
            normalized[:, :, i] = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
        return normalized
    else:
        # グレースケール画像の場合
        p_low = np.percentile(image, percentile_low)
        p_high = np.percentile(image, percentile_high)
        return np.clip((image - p_low) / (p_high - p_low), 0, 1)


def plot_tiepoints_on_images(
    optical_data: np.ndarray,
    sar_data: np.ndarray,
    tiepoints_df: pd.DataFrame,
    output_path: Path,
    num_samples: int = 5,
    window_size: int = 128
) -> None:
    """タイポイントを画像上にプロットして保存する"""
    
    # サンプルタイポイントを選択
    sample_indices = np.linspace(0, len(tiepoints_df) - 1, num_samples, dtype=int)
    sample_tiepoints = tiepoints_df.iloc[sample_indices]
    
    # 図の作成
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    # 各タイポイントについて処理
    for idx, (_, tiepoint) in enumerate(sample_tiepoints.iterrows()):
        opt_row = int(tiepoint['optical_row'])
        opt_col = int(tiepoint['optical_col'])
        sar_row = int(tiepoint['sar_row'])
        sar_col = int(tiepoint['sar_col'])
        
        # ウィンドウの範囲を計算
        half_window = window_size // 2
        
        # 光学画像のウィンドウ
        opt_row_start = max(0, opt_row - half_window)
        opt_row_end = min(optical_data.shape[0], opt_row + half_window)
        opt_col_start = max(0, opt_col - half_window)
        opt_col_end = min(optical_data.shape[1], opt_col + half_window)
        
        # SAR画像のウィンドウ
        sar_row_start = max(0, sar_row - half_window)
        sar_row_end = min(sar_data.shape[0], sar_row + half_window)
        sar_col_start = max(0, sar_col - half_window)
        sar_col_end = min(sar_data.shape[1], sar_col + half_window)
        
        # 光学画像のパッチを抽出
        optical_patch = optical_data[opt_row_start:opt_row_end, opt_col_start:opt_col_end]
        optical_patch_norm = normalize_image(optical_patch)
        
        # SAR画像のパッチを抽出
        sar_patch = sar_data[sar_row_start:sar_row_end, sar_col_start:sar_col_end]
        sar_patch_norm = normalize_image(sar_patch)
        
        # 光学画像をプロット
        ax_opt = axes[0, idx]
        if len(optical_patch_norm.shape) == 3:
            ax_opt.imshow(optical_patch_norm)
        else:
            ax_opt.imshow(optical_patch_norm, cmap='gray')
        
        # タイポイントの位置をマーク
        center_opt_row = opt_row - opt_row_start
        center_opt_col = opt_col - opt_col_start
        ax_opt.plot(center_opt_col, center_opt_row, 'r+', markersize=15, markeredgewidth=2)
        ax_opt.add_patch(plt.Circle((center_opt_col, center_opt_row), 10, 
                                   fill=False, edgecolor='red', linewidth=2))
        
        ax_opt.set_title(f'Optical\nPoint {idx+1}')
        ax_opt.axis('off')
        
        # SAR画像をプロット
        ax_sar = axes[1, idx]
        if len(sar_patch_norm.shape) == 3:
            ax_sar.imshow(sar_patch_norm)
        else:
            ax_sar.imshow(sar_patch_norm, cmap='gray')
        
        # タイポイントの位置をマーク
        center_sar_row = sar_row - sar_row_start
        center_sar_col = sar_col - sar_col_start
        ax_sar.plot(center_sar_col, center_sar_row, 'r+', markersize=15, markeredgewidth=2)
        ax_sar.add_patch(plt.Circle((center_sar_col, center_sar_row), 10, 
                                   fill=False, edgecolor='red', linewidth=2))
        
        ax_sar.set_title(f'SAR\n({sar_row}, {sar_col})')
        ax_sar.axis('off')
        
        # 対応する点を線で結ぶ
        con = ConnectionPatch(
            xyA=(center_opt_col, center_opt_row), coordsA=ax_opt.transData,
            xyB=(center_sar_col, center_sar_row), coordsB=ax_sar.transData,
            color='yellow', linewidth=2, alpha=0.5
        )
        fig.add_artist(con)
    
    plt.suptitle('Resampled Image Tiepoint Verification', fontsize=16)
    plt.tight_layout()
    
    # 保存
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"検証画像を保存しました: {output_path}")


def create_overview_plot(
    optical_data: np.ndarray,
    sar_data: np.ndarray,
    tiepoints_df: pd.DataFrame,
    output_path: Path
) -> None:
    """全体像の確認プロットを作成する"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 光学画像を表示
    optical_norm = normalize_image(optical_data)
    if len(optical_norm.shape) == 3:
        ax1.imshow(optical_norm)
    else:
        ax1.imshow(optical_norm, cmap='gray')
    
    # 光学画像上のタイポイントをプロット
    ax1.scatter(tiepoints_df['optical_col'], tiepoints_df['optical_row'], 
               c='red', s=20, alpha=0.6, marker='+')
    ax1.set_title(f'Optical Image\n({len(tiepoints_df)} tiepoints)')
    ax1.axis('off')
    
    # SAR画像を表示
    sar_norm = normalize_image(sar_data)
    if len(sar_norm.shape) == 3:
        ax2.imshow(sar_norm)
    else:
        ax2.imshow(sar_norm, cmap='gray')
    
    # SAR画像上のタイポイントをプロット
    ax2.scatter(tiepoints_df['sar_col'], tiepoints_df['sar_row'], 
               c='red', s=20, alpha=0.6, marker='+')
    ax2.set_title(f'SAR Image\n({len(tiepoints_df)} tiepoints)')
    ax2.axis('off')
    
    plt.suptitle('Resampled Images with All Tiepoints', fontsize=16)
    plt.tight_layout()
    
    # 保存
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"全体像を保存しました: {output_path}")


def process_region(region: str, config: Dict, base_dir: Path) -> None:
    """1つの地域のデータを処理する"""
    
    logging.info(f"地域 {region} の検証を開始します")
    
    # パスの設定
    interim_dir = base_dir / config['output']['interim_data_dir']
    resampled_dir = interim_dir / 'resampled' / region
    tiepoint_resampled_dir = interim_dir / 'tiepoint_resampled'
    verification_dir = interim_dir / 'resample_verification'
    
    optical_path = resampled_dir / '0_optical_resampled.tif'
    sar_path = resampled_dir / '1_sar_resampled.tif'
    tiepoints_path = tiepoint_resampled_dir / f'{region}_tiepoints_resampled.csv'
    
    # データを読み込む
    optical_data, optical_transform = load_resampled_image(optical_path)
    sar_data, sar_transform = load_resampled_image(sar_path)
    tiepoints_df = pd.read_csv(tiepoints_path)
    
    logging.info(f"光学画像サイズ: {optical_data.shape}")
    logging.info(f"SAR画像サイズ: {sar_data.shape}")
    logging.info(f"タイポイント数: {len(tiepoints_df)}")
    
    # 全体像をプロット
    overview_path = verification_dir / f'{region}_overview.png'
    create_overview_plot(optical_data, sar_data, tiepoints_df, overview_path)
    
    # サンプルタイポイントの詳細をプロット
    detail_path = verification_dir / f'{region}_tiepoint_details.png'
    plot_tiepoints_on_images(
        optical_data, sar_data, tiepoints_df, detail_path,
        num_samples=min(5, len(tiepoints_df))
    )


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='SpaceNet9 リサンプリング結果の検証')
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