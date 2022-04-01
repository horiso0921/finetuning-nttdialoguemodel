# finetuning-nttdialoguemodel

## 概要

このリポジトリには，NTTが提供する，日本語Transformer Encoder-decoder対話モデルを，[fairseq](https://github.com/pytorch/fairseq)上で学習するためのスクリプトが含まれています。


### 実行環境
- Python 3.8以降
- sentencepiece>=0.1.91
- torch>=1.5.1
- torchvision>=0.6.1

## 初学者向け
### 今すぐモデルの学習をさせてテストさせたい人へ（※入力データの成形は下の形式に沿って各自で行うこと）
1. docsのpdfにある「Jobスクリプト」より前まで行ってください（fairseqが/home/<id>にないと以降バグるので気を付けること）
1. `src`にて```bash all_setup.sh```をします
2. `data/RawData`の`/sample/`の6ファイルの中身を学習ソースに書き換えます
    1. 「train.src」：学習時のモデルへの入力データ（各行が1データにあたる）
        - 形式（\[SEP\]は発話単位の区切りトークン，\[SPK1\]と\[SPK2\]は話者トークンを表している）：```[SEP][SPK1]私は今日も元気[SEP][SPK2]私も元気[SEP][SPK1]それはよかった```
    2. 「train.dst」：学習時のモデルへの出力データ（各行が1データにあたる）
        - 形式（シンプルに発話だけ）：```うんよかった```
    3. 「valid.src」：検証時のモデルへの入力データ（各行が1データにあたる）
        - 形式：train.srcと同じ
    4. 「valid.dst」：検証時のモデルへの出力データ（各行が1データにあたる）
        - 形式：train.dstと同じ 
    5. 「test.src」：テスト時のモデルへの入力データ（各行が1データにあたる）
        - 形式：train.srcと同じ
    6. 「test.dst」：テスト時のモデルへの出力データ（各行が1データにあたる）
        - 形式：train.dstと同じ    
3. `job`にて```pjsub finetune.sh```を実行
    - `pjstat` で Jobが走っているか確認できる
    - `job/sample/` にログがいろいろ生えるので適当にみる。
    - `job/sample/test_<timestamp>.log`で結果を確認できる

## 中身をちょっと知りたい人向け
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
                - 学習データの入力データファイル名：train.src
                - 学習データの出力データファイル名：train.dst
                - 検証データの入力データファイル名：valid.src
                - 学習データの入力データファイル名：valid.dst
        - PreprocessedData
            - 事前処理をしてトークナイズされたデータを入れるところです
            - トークナイズがきちんとされているか確認してください
            - サブディレクトリごとにデータを分けると便利です
        - PreprocessedBinaryData
            - バイナリ化された事前処理をしてトークナイズされたデータを入れるところです
            - 人間には解読不可能です。
            - 一度処理をしてこのディレクトリにdict.src.txtが生成されてしまうと事前処理が走らなくなるので注意してください
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
