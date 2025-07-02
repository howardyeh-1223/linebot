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
            model="gpt-4o",
           messages=[
            {
                 "role": "system",
                 "content": "你是一個只會翻譯語言的工具，請依照下列規則翻譯：\n"
                            "- 如果輸入是中文，請翻譯成印尼文。\n"
                            "- 如果輸入是印尼文，請翻譯成繁體中文。\n"
                            "- 不要加任何說明、解釋、問候語。\n"
                            "- 只輸出翻譯結果。\n"
                            "- 所有中文回覆都必須使用繁體中文，禁止使用簡體字。"           
            },
            {
                "role": "user",
                "content": text
            }
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
