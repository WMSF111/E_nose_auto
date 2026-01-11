# app.py
import webbrowser
from flask import Flask, render_template_string
from dotenv import load_dotenv
import os
import threading

# 加载 .env 文件中的环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)

@app.route('/')
def index():
    # 从环境变量中读取 token 和 appId
    token = os.getenv('COZE_TOKEN')
    app_id = os.getenv('COZE_APPID')

    # HTML模板字符串，动态插入 token 和 appId
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Web SDK Example</title>
        <style>
          body {{
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f0f0f0;
          }}
          #app {{
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #fff;
            border: 1px solid #ddd;
          }}
        </style>
    </head>
    <body>
        <div id="app"></div>

        <script type="text/javascript">
          var webSdkScript = document.createElement('script');
          webSdkScript.src = 'https://lf-cdn.coze.cn/obj/unpkg/flow-platform/builder-web-sdk/0.1.1-beta.1/dist/umd/index.js';
          document.head.appendChild(webSdkScript);

          webSdkScript.onload = function () {{
            new CozeWebSDK.AppWebSDK({{
              "token": "{token}",
              "appId": "{app_id}",
              "container": "#app",
              "userInfo": {{
                "id": "User",
                "url": "https://lf-coze-web-cdn.coze.cn/obj/eden-cn/lm-lgvj/ljhwZthlaukjlkulzlp/coze/coze-logo.png",
                "nickname": "User"
              }}
            }});
          }}
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

def run():
    # 启动 Flask 应用
    app.run(debug=False)

def open_browser():
    # 在浏览器中打开 Flask 应用
    webbrowser.open("http://127.0.0.1:5000")

# 在主函数中，使用线程来启动浏览器和 Flask 应用
if __name__ == '__main__':
    # 创建一个线程来启动 Flask 应用
    flask_thread = threading.Thread(target=run)
    flask_thread.start()

    # 启动浏览器
    open_browser()
