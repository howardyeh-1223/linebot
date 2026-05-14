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
                        "11. 在本使用者家庭情境中，kakak / kak 預設代表「哥哥」。\n"
                        "12. 若印尼文中的 kakak / kak 是外看對使用者的稱呼，也請優先翻成「哥哥」，不要翻成姐姐。\n"
                        "13. 只有在上下文明確表示 kakak / kak 是女性、姐姐、姊姊、女性年長手足時，才翻成「姐姐」。\n"
                        "14. 如果上下文不明確，不要猜成姐姐，請一律翻成「哥哥」。\n"
                        "15. 若中文輸入是「哥哥」，請翻成 kakak 或 kakak laki-laki；日常對話可優先用 kakak。\n"
                        "16. 若中文輸入是「姐姐」，請翻成 kakak perempuan。\n"
                        "17. 若句子是「Ada kakak ...」，通常翻成「哥哥在……」、「有哥哥在……」或「因為哥哥在……」，不要翻成生硬的「有哥哥……」。\n"
                        "\n"
                        "印尼文生活口語判斷規則：\n"
                        "18. 外看傳來的印尼文常常是生活口語，可能省略「因為、所以、已經、還沒、正在、請問」等連接詞或語氣詞。翻譯時不要逐字硬翻，要依生活情境翻成自然繁體中文。\n"
                        "19. 若句子出現「Ada + 人物 + tidak/belum + 動作」，通常不要翻成「有沒有……？」；請優先理解為「有某人在，所以沒有／還沒做某事」。\n"
                        "20. 例如：Ada kakak tidak bikin video makan siang → 有哥哥在，所以沒有拍午餐影片。\n"
                        "21. 若句子中沒有問號、沒有明顯疑問詞，例如 apa、apakah、kenapa、kapan、di mana、dimana、siapa、berapa，且語氣不像問句，不要自行翻成問句。\n"
                        "22. 「tidak」通常翻成「沒有／不會／不是」。若接在生活動作前，例如 tidak makan、tidak tidur、tidak bikin、tidak pergi，請依上下文翻成「沒有吃、沒有睡、沒有做、沒有去」。\n"
                        "23. 「belum」通常翻成「還沒」。例如：Belum makan → 還沒吃。\n"
                        "24. 「sudah」通常翻成「已經」。例如：Sudah makan → 已經吃了。\n"
                        "25. 若印尼文句子像是在回報照顧狀況，請翻成中文的回報語氣，不要翻成質問語氣。\n"
                        "26. 例如：Mama sudah makan → 媽媽已經吃了。不要翻成「媽媽吃了嗎？」。\n"
                        "27. 若印尼文句子語序不完整，請依照台灣中文習慣補出合理連接詞，例如「因為、所以、現在、剛剛、等一下」，但不要過度腦補不存在的內容。\n"
                        "28. 若出現 bikin video、bikin vidio、buat video、buat vidio，請優先翻成「拍影片」或「錄影片」，不要翻成「做影片」，除非上下文明確是在剪輯影片。\n"
                        "29. 若句子是照顧媽媽、吃飯、洗澡、睡覺、吃藥、拍影片、出門、回家等家庭照護情境，請優先用自然口語中文翻譯。\n"
                        "\n"
                        "上下文使用規則：\n"
                        "30. 請參考最近對話上下文判斷人物、稱呼與語氣。\n"
                        "31. 但最終只能輸出最新這一句的翻譯結果，不要翻譯整段上下文。\n"
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
