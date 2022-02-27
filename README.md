# finetuning-nttdialoguemodel

## 概要

このリポジトリには，NTTが提供する，日本語Transformer Encoder-decoder対話モデルを，[fairseq](https://github.com/pytorch/fairseq)上で学習するためのスクリプトが含まれています。


### 利用方法
#### 実行環境
- Python 3.8以降
- sentencepiece>=0.1.91
- torch>=1.5.1
- torchvision>=0.6.1


### 共有されているモデルをどこに置くか
- 学習データ
    - 下のディレクトリ構造を参照
- モデル
    - 事前学習済応答生成モデル
        - `model/base` に置く
        - `1.6B_2lhzhoam_4.92.pt`か`empdial50k-flat_1.6B_19jce27w_3.86.pt`か`persona50k-flat_1.6B_33avog1i_4.16.pt`を置けばよいです
        - 使用するモデル名を`setting_fine_tuning.sh`で変更するようにしてください

    - spm
        - `model/spm` に置く
        - sp_oall_32k.{model, txt, vocab}をすべて置いてください

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
        - _prepro_spm.py
            - 生データに存在するデータすべてを対象にセンテンスピースを使ってトークナイズするスクリプトです
            - 不老のJobから呼び出せないので注意
        - _prepro_spm_simple.py
            - 第一引数で受け取った名前の生データを対象にセンテンスピースを使ってトークナイズするスクリプトです
            - **SPM，SRC_DIR，DATA_DIRのPathを必ず変えること**
        - preprocess.sh
            - 事前処理されたデータをバイナリにするスクリプトです
            - **PRE_DATA_DIR，AFTER_DATA_DIR，SPM_VOCAB のPathを必ず変えること**
        - setting_fine_tuning.sh
            - 学習用のパラメータを少しまとめたスクリプトです
            - モデル構造等を書きます
            - **WORK_ROOT_DIRのPathを必ず変えること**
        - train_fine_tuning.sh
            - 学習を実行するスクリプトです
            - 学習時のパラメータはここで調節します
            - **preprocess.sh と setting_fine_tuning.sh の Pathは必ず変えること**
        - train_sentencepiece_model.py
            - トークナイザーを学習するスクリプトです
            - メンテナンスしていないので動くか不明です

### 学習方法の詳細
1. _prepro_spm.pyを実行
2. bash preproccess_fairseq.shを実行（**任意**）
    - 第一引数にデータ名を入れること（RawDataのサブDirの名前）
3. bash train_fine_tuning.shを実行
    - 第一引数から順にデータ名，Baseにするモデル（BASE, EMP, PERのみ対応），WARMUP_STEP，BATCH_SIZE，LRを入れること

### Tenosorboard
`tensorboard --logdir .`
