# dockerを用いて、aws lambdaにpythonのライブラリを.zip形式でアップロードし、importさせる方法

1. getting-started ディレクトリに移動する
2. requirements.txtに必要なライブラリを記述
3. docker run --rm -v $(pwd):/var/task lambci/lambda:build-python3.6 pip install -r requirements.txt -t python/lib/python3.6/site-packages/
4. zip -r 名前.zip ./python > /dev/null
5. 名前.zipをaws lambdaにアップロードする