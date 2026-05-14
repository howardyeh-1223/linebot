from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict, deque
import os
import logging

# ============================================================
# 基本設定
# ============================================================

# 本機測試時會讀 .env
# Render 上正式執行時，會讀 Render Environment Variables
load_dotenv()

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 可在 Render 的 Environment Variables 設定
# 建議先用 gpt-4.1-mini，若覺得品質還不夠，再改成 gpt-4.1
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

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
# 對話記憶
# ============================================================

# 每個 LINE user 保留最近 10 筆原文與翻譯
# Render 重新部署或休眠後，這個記憶會消失
conversation_memory = defaultdict(lambda: deque(maxlen=10))


# ============================================================
# 翻譯邏輯：中文 ↔ 印尼文
# ============================================================

def translate_text(user_id: str, text: str) -> str:
    """
    中文翻譯成印尼文。
    印尼文翻譯成繁體中文。

    特別規則：
    - kakak / kak 預設翻成「哥哥」。
    - 除非上下文明確是在說女性年長手足，才翻成「姐姐」。
    - 若 kakak / kak 是外看對使用者的稱呼，也優先翻成「哥哥」。
    """

    if not text or not text.strip():
        return ""

    user_text = text.strip()

    # 取出最近對話上下文
    recent_messages = list(conversation_memory[user_id])

    if recent_messages:
        context_text = "\n".join(
            [f"{item['role']}：{item['content']}" for item in recent_messages]
        )
    else:
        context_text = "目前沒有上下文。"

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
                        "\n"
                        "關於 kakak / kak 的特殊規則：\n"
                        "11. 使用者的家庭情境中，kakak / kak 預設翻成「哥哥」。\n"
                        "12. 若印尼文中的 kakak / kak 是外看對使用者的稱呼，也請優先翻成「哥哥」，不要翻成姐姐。\n"
                        "13. 只有在上下文明確表示 kakak / kak 是女性、姐姐、姊姊、女性年長手足時，才翻成「姐姐」。\n"
                        "14. 如果上下文不明確，不要猜成姐姐，請一律翻成「哥哥」。\n"
                        "15. 若中文輸入是「哥哥」，請翻成 kakak 或 kakak laki-laki；日常對話可優先用 kakak。\n"
                        "16. 若中文輸入是「姐姐」，請翻成 kakak perempuan。\n"
                        "\n"
                        "上下文使用規則：\n"
                        "17. 請參考最近對話上下文判斷人物、稱呼與語氣。\n"
                        "18. 但最終只能輸出最新這一句的翻譯結果，不要翻譯整段上下文。"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "以下是最近對話上下文，僅供判斷稱呼、人物與語氣，不要翻譯整段上下文：\n"
                        f"{context_text}\n\n"
                        "請只翻譯最新這一句：\n"
                        f"{user_text}"
                    )
                }
            ]
        )

        translated = response.choices[0].message.content

        if not translated:
            return "翻譯失敗：沒有取得翻譯結果"

        translated = translated.strip()

        # 儲存原文與翻譯，供下一句參考
        conversation_memory[user_id].append({
            "role": "原文",
            "content": user_text
        })

        conversation_memory[user_id].append({
            "role": "翻譯",
            "content": translated
        })

        return translated

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

    # LINE user id，用來區分不同人的上下文
    user_id = event.source.user_id

    translated = translate_text(user_id, user_message)

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
