from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
app = Flask(__name__)
# 讀取 run.env 中的環境變數
load_dotenv(dotenv_path='run.env')

# 讀取變數
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 初始化
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
openai.api_key = OPENAI_API_KEY
app = Flask(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)
# 翻譯邏輯：中↔印
def translate_text(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "你是一個只負責翻譯的工具。"
                               "請將輸入的文字翻譯成印尼文（如果是中文），"
                               "或翻譯成中文（如果是印尼文）。"
                               "不要回應問題、不要回答、不要多講，只輸出翻譯後的文字。"
                               "中文請回繁體中文。"
                },
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"翻譯失敗：{str(e)}"

@app.route("/", methods=["GET"])
def home():
    return "LINE Bot 已上線"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    translated = translate_text(user_message)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=translated)
    )

if __name__ == "__main__":
    app.run()
