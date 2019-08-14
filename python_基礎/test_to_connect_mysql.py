import mysql.connector

conn = mysql.connector.connect(user='root', password='', host='localhost', database='freemarket_sample_36a_development')
cur = conn.cursor()

cur.execute("select * from users;")

for row in cur:
    print(row[0],row[1])

cur.close
conn.close
