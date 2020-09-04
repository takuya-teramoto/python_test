#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:42:43 2020

@author: takuya.teramoto
"""

import os
import numpy as np
import pandas as pd
import sqlalchemy as sqa
import json
import websocket

# mysplの基本設定
user = 'root'
password = os.environ['MYSQL_PASSWORD']
host = 'localhost'
database = 'CryptoCurrency'
table_name = 'GMO'

# urlの定義(mysql)
url = "mysql+pymysql://{}:{}@{}/{}".format(user, password, host, database)
engine = sqa.create_engine(url, echo=True)

# GMOから仮想通貨の情報を取得する
# ソケット通信の初期設定
websocket.enableTrace(True)
ws = websocket.WebSocketApp('wss://api.coin.z.com/ws/public/v1')

def on_open(self):
    message = {
        "command": "subscribe",
        "channel": "ticker",
        "symbol": "BTC_JPY"
    }
    ws.send(json.dumps(message))

# callback関数
def on_message(self, message):
    # stringで取得したmessageをdfに変換
    json_message = json.loads(message)
    df_message = pd.DataFrame([json_message])
    # mysqlに書き込み
    df_message.to_sql(table_name, url, index=None, if_exists='append')

ws.on_open = on_open
ws.on_message = on_message

ws.run_forever()

# # 読み込み
# sql = """
#         SELECT
#             *
#         FROM
#             {}
# """.format(table_name)

# df_read = pd.read_sql(sql, url)
