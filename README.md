# Copyright (c) 2022 Katori lab. All Rights Reserved

# cbm_rc_objectoriented

cbm_rc が煩雑になってきたので整理のためにオブジェクト指向フォルダを作成します。
使い方の一例はmemory2.pyを見てください。

# explorer
探索アルゴリズムなどを行う。
# generate_dataset
各タスクのデータ生成を行う、
# __init__.py
上位からの呼び出し時に　_network.pyのクラスのimportを行う。
# _network.py
RCCBMのクラス表現。
# generate_matrix.py
各荷重行列の生成を行う。
# utils.py
使用する関数置き場。


