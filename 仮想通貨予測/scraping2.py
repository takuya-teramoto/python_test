
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:42:43 2020

@author: takuya.teramoto
"""

import os
import pandas as pd
import sqlalchemy as sqa
import json
import requests

# mysplの基本設定
user = 'admin'
password = os.environ['MYSQL_PASSWORD']
host = 'cryptocurrency.cinqcdge0puy.ap-northeast-1.rds.amazonaws.com'
database = 'cryptocurrency'
table_name = 'GMO'

# urlの定義(mysql)
url = "mysql+pymysql://{}:{}@{}/{}".format(user, password, host, database)
engine = sqa.create_engine(url, echo=True)

# GMOから仮想通貨の情報を取得する
# ソケット通信の初期設定
endPoint = 'https://api.coin.z.com/public'
path     = '/v1/ticker?symbol=BTC'

response = requests.get(endPoint + path)
json_response = json.loads(json.dumps(response.json(), indent=2))
data = json_response['data'][0]
df = pd.DataFrame([data])
print(df)
# mysqlに書き込み
# df.to_sql(table_name, url, index=None, if_exists='append')

