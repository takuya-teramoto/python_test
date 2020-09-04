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
user = 'admin'
password = os.environ['MYSQL_PASSWORD']
host = 'cryptocurrency.cinqcdge0puy.ap-northeast-1.rds.amazonaws.com'
database = 'cryptocurrency'
table_name = 'GMO'

# urlの定義(mysql)
url = "mysql+pymysql://{}:{}@{}/{}".format(user, password, host, database)
engine = sqa.create_engine(url, echo=True)


# 読み込み
sql = """
        SELECT
            *
        FROM
            {}
""".format(table_name)

df_read = pd.read_sql(sql, url)
df_read.to_csv(os.path.join(os.getcwd(), 'all.csv'), index=False)
