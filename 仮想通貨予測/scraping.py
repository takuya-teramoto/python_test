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


user = 'root'
password = os.environ['MYSQL_PASSWORD']
host = 'localhost'
database = 'CryptoCurrency'
table_name = 'test'
url = "mysql+pymysql://{}:{}@{}/{}".format(user, password, host, database)

df = pd.DataFrame(np.arange(12).reshape(-1, 2))
df.columns = ['test1', 'test3']

# 書き込み
engine = sqa.create_engine(url, echo=True)
df.to_sql(table_name, url, index=None, if_exists='append')


# 読み込み
sql = """
        SELECT
            *
        FROM
            {}
""".format(table_name)

df_read = pd.read_sql(sql, url)
