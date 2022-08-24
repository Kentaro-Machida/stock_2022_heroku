import sys
sys.path.append("./src")

import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
import cv2
from src.stock_detection import get_detected_image

app = Flask(__name__)
app.secret_key = "test"

# パラメータ設定ファイルへのパス
CONFIG_PATH = "./roi_config.ini"

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './uploads'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif','jpeg','JPG','JPEG','PNG'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def uploads_file():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['file']
        # ファイル名がなかった時の処理
        if file.filename == '':
            return redirect(request.url)
        # ファイルのチェック
        if file and allwed_file(file.filename):
            # 危険な文字を削除（サニタイズ処理）
            filename = secure_filename(file.filename)
            # ファイルの保存
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
            file.save(img_path)
            # アップロード後のページに転送
            detected ,detected_img = get_detected_image(config_path = CONFIG_PATH,img_path=img_path)
            if detected == False:
                return """
                ストックが発見できませんでした。以下のことがないかを確認してください。\n
                ・ストックとの距離が近すぎる。遠すぎる。
                ・葉どうしが重なってしまっている。
                ・葉の色と似た物体が一緒に写り込んでしまっている。
                """
            
            out_path = os.path.join("detected", filename)
            cv2.imwrite(out_path, detected_img)
            return redirect(url_for('uploaded_file', filename=filename))
    return '''
    <!doctype html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>
                ストックの画像をアップロードして八重鑑別しよう！
            </title>
        </head>
        <body>
            <h1>
                ストックの画像をアップロードして八重鑑別しよう！
            </h1>
            <form method = post enctype = multipart/form-data>
            <p><input type=file name = file>
            <input type = submit value = Upload>
            </form>
        </body>
'''

@app.route('/uploads/<filename>')
# ファイルを表示する
def uploaded_file(filename):
    return send_from_directory("detected", filename)

if __name__=='__main__':
    app.run()