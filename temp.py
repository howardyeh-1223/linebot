from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv
from openai import OpenAI
import os
import logging

# ============================================================
# 基本設定
# ============================================================

app = Flask(__name__)

# 讀取本機 run.env
# Render 上通常不用 run.env，而是直接在 Environment 設定環境變數
load_dotenv(dotenv_path="run.env")

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 模型可由環境變數控制
# 建議：
# 1. 成本與速度平衡：gpt-4.1-mini
# 2. 想更強可以在 Render 改成：gpt-5.4-mini
# 3. 若要最高品質且不太在意成本：gpt-5.5
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# 檢查必要環境變數
required_env = {
    "LINE_CHANNEL_ACCESS_TOKEN": LINE_CHANNEL_ACCESS_TOKEN,
    "LINE_CHANNEL_SECRET": LINE_CHANNEL_SECRET,
    "OPENAI_API_KEY": OPENAI_API_KEY,
}

missing = [key for key, value in required_env.items() if not value]
if missing:
    raise RuntimeError(f"缺少必要環境變數：{', '.join(missing)}")

# 初始化 LINE / OpenAI
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
client = OpenAI(api_key=OPENAI_API_KEY)

# Log 設定
logging.basicConfig(level=logging.INFO)


# ============================================================
# 翻譯邏輯：中文 ↔ 印尼文
# ============================================================

def translate_text(text: str) -> str:
    """
    中文翻譯成印尼文。
    印尼文翻譯成繁體中文。
    只輸出翻譯結果，不加說明。
    """

    if not text or not text.strip():
        return ""

    user_text = text.strip()

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一個高品質的中文與印尼文雙向翻譯工具。\n"
                        "\n"
                        "請嚴格遵守以下規則：\n"
                        "1. 如果使用者輸入主要是中文，請翻譯成自然、口語、符合印尼人日常使用習慣的印尼文。\n"
                        "2. 如果使用者輸入主要是印尼文，請翻譯成自然、口語、符合台灣用法的繁體中文。\n"
                        "3. 中文輸出一律使用繁體中文，禁止使用簡體中文。\n"
                        "4. 只輸出翻譯結果，不要加任何說明、註解、前綴、問候語或引號。\n"
                        "5. 不要輸出「印尼翻譯是」、「中文翻譯是」、「以下是翻譯」這類文字。\n"
                        "6. 保留原文語氣，例如：撒嬌、抱怨、禮貌、命令、疑問、簡短回覆，都要盡量維持。\n"
                        "7. 專有名詞、人名、地名、品牌名稱，如果不確定，不要亂翻，可保留原文。\n"
                        "8. 表情符號、數字、日期、金額、網址、電話、代碼請保留原格式。\n"
                        "9. 如果輸入內容混合中文與印尼文，請判斷主要語言，翻譯成另一種語言。\n"
                        "10. 如果內容太短，例如「好」、「嗯」、「可以」、「iya」、「ok」，請翻成自然對應語氣，不要過度延伸。\n"
                    )
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ]
        )

        translated = response.choices[0].message.content

        if not translated:
            return "翻譯失敗：沒有取得翻譯結果"

        return translated.strip()

    except Exception as e:
        logging.exception("OpenAI 翻譯失敗")
        return f"翻譯失敗：{str(e)}"


# ============================================================
# Render / LINE Webhook
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return "LINE Bot 已上線"


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    if not signature:
        abort(400)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception:
        logging.exception("LINE callback 發生錯誤")
        abort(500)

    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    translated = translate_text(user_message)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=translated)
    )


# ============================================================
# 本機測試用
# Render 通常會用 gunicorn app:app 啟動
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
