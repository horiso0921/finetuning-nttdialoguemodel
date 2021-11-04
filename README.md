# finetuning-nttdialoguemodel

## 概要

このリポジトリには，NTTが提供する，日本語Transformer Encoder-decoder対話モデルを，[fairseq](https://github.com/pytorch/fairseq)上で学習するためのスクリプトが含まれています。


### 利用方法
#### 実行環境
- Python 3.8以降
- sentencepiece>=0.1.91
- torch>=1.5.1
- torchvision>=0.6.1

fairseqについては，このリポジトリのクローン後，以下の手順で導入してください
```
cd finetuning-nttdialoguemodel/fairseq
pip install --editable .
```

### ディレクトリ構造
- data
    - 学習に必要なデータを格納するディレクトリです
    - Subdir
        - RawData
            - 事前処理する前のコードを入れるところです
            - サブディレクトリごとにデータを分けると便利です
            - <b>以下の規則を必ず守ってください</b>
                - 学習データの入力データファイル名：train.context
                - 学習データの出力データファイル名：train.response
                - 検証データの入力データファイル名：valid.context
                - 学習データの入力データファイル名：valid.response
        - PreprocessedData
            - 事前処理をしてトークナイズされたデータを入れるところです
            - トークナイズがきちんとされているか確認してください
            - サブディレクトリごとにデータを分けると便利です
        - PreprocessedBinaryData
            - バイナリ化された事前処理をしてトークナイズされたデータを入れるところです
            - 人間には解読不可能です。
            - 一度処理をしてこのディレクトリにdict.context.txtが生成されてしまうと事前処理が走らなくなるので注意してください
            - 更にサブディレクトリごとにデータを分けると便利です

- src
    - 学習や事前準備に必要なコードがまとまっています
        - prepro_spm.py
            - 生データを事前処理してセンテンスピースを使ってトークナイズするスクリプトです
        - preprocess_fairseq.sh
            - 事前処理されたデータをバイナリにするスクリプトです
        - setting_fine_tuning.sh
            - 学習用のパラメータを少しまとめたスクリプトです
            - モデル構造等を書きます
        - train_fine_tuning.sh
            - 学習を実行するスクリプトです
            - 学習時のパラメータはここで調節します
        - train_sentencepiece_model.py
            - トークナイザーを学習するスクリプトです
            - メンテナンスしていないので動くか不明です

### 学習方法の詳細