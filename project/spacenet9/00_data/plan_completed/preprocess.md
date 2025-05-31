# SpaceNet9 データ前処理計画

## 概要
光学画像とSAR画像を同じAOI（Area of Interest）にタイル化し、対応するタイポイントを計算して、処理済みデータとして保存する。

## 目的
- 大規模な衛星画像を扱いやすいサイズ（256px、512px等）にタイル化
- 光学画像とSAR画像の対応するタイルを生成
- 各タイルに含まれるタイポイントを抽出・変換
- 学習・推論に適した形式でデータを保存

## ディレクトリ構造

```
project/spacenet9/00_data/
├── raw/                    # 元データ
├── interim/                # 中間データ
├── dataset/                # 前処理済みデータ
├── plan/                   # 前処理要件定義ドキュメント
└── plan_completed/         # 前処理要件定義ドキュメントの完了したファイル
```

## 前処理スクリプト


```
project/spacenet9/src/
├── preprocess/            #各処理ごとにスクリプトを作成
└── test/                  #各処理のテストスクリプトを作成 
```


## TODO リスト

### 1. Tiepointのピクセル値を経緯度座標に変換
- [x] pre_alignment.pyをsrc/preprocess/ディレクトリに作成
- [x] GeoTIFFメタデータから座標変換行列を取得
- [x] 座標変換行列を適用してTiepointのピクセル値を経緯度座標に変換
- [x] interim/tiepoint_converted ディレクトリに変換後のTiepointファイルを対象領域ごとにcsv形式で保存

### 2. 光学画像とSAR画像のリサンプル
- [x] 光学画像とSAR画像を重複するAOIで切り取り、同じ解像度になるようにリサンプル
- [x] リサンプル後の光学画像とSAR画像をinterim/resampledディレクトリに保存
- [x] Tiepointをリサンプル後のピクセルに対応するように変換
   - [x] raw/train/*/2_tiepoints.csvと同じ形式になるように情報を追加・削除
- [x] Tiepointをinterim/tiepoint_resampledディレクトリに保存

### 3. リサンプル後の画像とTiepointの確認
- [x] 確認用にいくつかの対応するTiepointを抽出して、リサンプル後のSAR画像と光学画像に重ね合わせて表示
- [x] 重ね合わせた画像をinterim/resample_verificationディレクトリに保存

### 4. 光学画像とSAR画像の対応するタイルを生成
- [x] 光学キーポイント検出データセットの生成
   - [x] SAR画像の各Tiepointを中心に光学およびSAR画像を256px × 256pxにタイル化
   - [x] 各タイルに地理情報（CRS、変換行列）を付与
   - [x] dataset/opt_keypoint_detectionディレクトリに光学キーポイント検出データセットを各シーンごとに保存
   - [x] ラベル画像として上で使用したSAR画像Tiepointに対応する光学画像のTiepointを中心に分散=13の正規分布にしたがうキーポイントを生成
   - [x] dataset/opt_keypoint_detectionディレクトリに光学キーポイント検出データセットを各シーンごとに保存
- [x] SARキーポイント検出データセットの生成
   - [x] 光学画像の各Tiepointを中心に光学およびSAR画像を256px × 256pxにタイル化
   - [x] 各タイルに地理情報（CRS、変換行列）を付与
   - [x] dataset/sar_keypoint_detectionディレクトリにSARキーポイント検出データセットを各シーンごとに保存
   - [x] ラベル画像として上で使用した光学画像Tiepointに対応するSAR画像のTiepointを中心に分散=13の正規分布にしたがう値を持つように生成
   - [x] dataset/sar_keypoint_detectionディレクトリにSARキーポイント検出データセットを各シーンごとに保存

### 5. 各処理のスクリプトのテスト作成・実行
- [x] preprocessディレクトリにある各処理をテストするスクリプトをpytestで作成
- [x] テストスクリプトを実行・エラーがないことを確認