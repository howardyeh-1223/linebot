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
# Render 正式執行時，會讀 Render 的 Environment Variables
load_dotenv()

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 可在 Render Environment Variables 設定
# 建議先用 gpt-4.1-mini
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
# 對話記憶
# ============================================================

# 每個 LINE user 保留最近 10 筆原文與翻譯
# 注意：Render 重新部署、重啟、休眠後，這個記憶會消失
conversation_memory = defaultdict(lambda: deque(maxlen=10))


# ============================================================
# 翻譯邏輯：中文 ↔ 印尼文
# ============================================================

def translate_text(user_id: str, text: str) -> str:
    """
    中文翻譯成印尼文。
    印尼文翻譯成繁體中文。

    特色：
    1. 針對台灣家庭與印尼外看的生活對話最佳化。
    2. kakak / kak 預設翻成「哥哥」。
    3. 會參考同一個 LINE 使用者最近幾句對話。
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
                        "你是一個專門處理台灣家庭與印尼外看之間對話的中文與印尼文雙向翻譯工具。\n"
                        "\n"
                        "翻譯目標：\n"
                        "1. 中文翻成自然、口語、符合印尼外看日常使用習慣的印尼文。\n"
                        "2. 印尼文翻成自然、口語、符合台灣家庭日常用法的繁體中文。\n"
                        "3. 只輸出翻譯結果，不要加說明、註解、前綴、問候語或引號。\n"
                        "4. 中文一律使用繁體中文，禁止使用簡體中文。\n"
                        "5. 不要輸出「印尼翻譯是」、「中文翻譯是」、「以下是翻譯」這類文字。\n"
                        "\n"
                        "重要情境：\n"
                        "6. 對話多半發生在家庭照護情境。\n"
                        "7. 常見內容包含：奶奶、媽媽、哥哥、外看、吃飯、睡覺、洗澡、吃藥、出門、回家、拍影片、錄影片、回報狀況。\n"
                        "8. 印尼文常常是生活口語，可能文法不完整、拼字不標準、沒有標點、語序不完整。\n"
                        "9. 外看可能會省略「因為、所以、已經、還沒、正在、要不要、請問」等連接詞或語氣。\n"
                        "10. 翻譯時請根據家庭照護情境理解意思，不要逐字硬翻。\n"
                        "11. 翻譯前請先在內部判斷這句是「回報、詢問、請求、提醒、抱怨、說明」哪一類，但不要輸出判斷過程，只輸出翻譯。\n"
                        "\n"
                        "語氣判斷：\n"
                        "12. 如果一句話不像問句，不要硬翻成問句。\n"
                        "13. 如果像是在回報狀況，請翻成回報語氣。\n"
                        "14. 如果像是在詢問，請翻成自然問句。\n"
                        "15. 如果像是在提醒，請翻成自然提醒語氣。\n"
                        "16. 如果內容很短，例如「好」、「嗯」、「可以」、「iya」、「ok」，請翻成自然對應語氣，不要過度延伸。\n"
                        "\n"
                        "固定稱呼：\n"
                        "17. kakak / kak 在本使用者家庭情境中，預設翻成「哥哥」。\n"
                        "18. 若印尼文中的 kakak / kak 是外看對使用者的稱呼，也請優先翻成「哥哥」。\n"
                        "19. 只有上下文明確表示 kakak / kak 是女性、姐姐、姊姊、女性年長手足時，才翻成「姐姐」。\n"
                        "20. 如果上下文不明確，不要猜成姐姐，請一律翻成「哥哥」。\n"
                        "21. 中文輸入「哥哥」時，請翻成 kakak；若需要強調男性，才翻成 kakak laki-laki。\n"
                        "22. 中文輸入「姐姐」時，請翻成 kakak perempuan。\n"
                        "23. tuan 在外看對使用者說話時，翻成「先生」或自然省略。\n"
                        "24. 如果同一句裡 tuan 出現多次，中文只保留一次即可，不要重複翻成「先生、先生」。\n"
                        "\n"
                        "生活口語理解：\n"
                        "25. 印尼文的 ada 在生活對話裡常表示「在、有、正在、因為某人在場」。請依上下文翻成自然中文。\n"
                        "26. 若句子像「Ada + 人物 + tidak/belum + 動作」，通常不要翻成「有沒有……？」；請優先理解為「有某人在，所以沒有／還沒做某事」。\n"
                        "27. 若沒有問號，也沒有明顯疑問詞，例如 apa、apakah、kenapa、kapan、di mana、dimana、siapa、berapa，且語氣不像問句，不要自行翻成問句。\n"
                        "28. tidak 接生活動作時，常翻成「沒有……」、「不……」。例如沒有吃、沒有睡、沒有去、沒有拍。\n"
                        "29. belum 通常翻成「還沒」。\n"
                        "30. sudah 通常翻成「已經」。\n"
                        "31. lagi / sedang 常表示「正在」。\n"
                        "32. mau 常表示「要、想要、準備要」。\n"
                        "33. nanti 常表示「等一下、晚一點」。\n"
                        "34. sekarang 常表示「現在」。\n"
                        "\n"
                        "家庭照護常見詞：\n"
                        "35. Nenek / nenek 請依情境翻成「奶奶」。\n"
                        "36. Mama / mama 請依情境翻成「媽媽」。\n"
                        "37. makan 請依情境翻成「吃飯」或「吃」。\n"
                        "38. makan siang 請翻成「午餐」或「吃午餐」。\n"
                        "39. tidur 請依情境翻成「睡了」、「在睡覺」、「睡覺」。\n"
                        "40. mandi 請翻成「洗澡」。\n"
                        "41. obat 請翻成「藥」。\n"
                        "42. minum obat 請翻成「吃藥」或「喝藥」，台灣中文優先用「吃藥」。\n"
                        "43. jalan keluar、mau jalan keluar、apakah jalan keluar 在家庭照護對話中，通常理解為「出門／出去」，不要逐字翻成「出口」或「出路」。\n"
                        "44. 只有在明確是在問建築物、道路、逃生、門口方向時，jalan keluar 才翻成「出口」或「出路」。\n"
                        "45. bikin video、bikin vidio、buat video、buat vidio 請優先翻成「拍影片」或「錄影片」，不要翻成「做影片」，除非上下文明確是在剪輯影片。\n"
                        "\n"
                        "代表性例句，請學習其語感，不要輸出例句：\n"
                        "46. Ada kakak tidak bikin vidio makan siang → 有哥哥在，所以沒有拍午餐影片。\n"
                        "47. Nenek tidur tuan apakah jalan keluar tuan → 奶奶睡了，先生要出門嗎？\n"
                        "48. Mama sudah makan → 媽媽已經吃了。\n"
                        "49. Belum makan → 還沒吃。\n"
                        "50. Nenek belum tidur → 奶奶還沒睡。\n"
                        "51. Mama lagi tidur → 媽媽正在睡覺。\n"
                        "52. Tuan mau keluar? → 先生要出門嗎？\n"
                        "53. Kakak sudah makan? → 哥哥吃飯了嗎？\n"
                        "\n"
                        "上下文規則：\n"
                        "54. 請參考最近對話上下文判斷人物、稱呼、語氣和省略內容。\n"
                        "55. 但最終只能輸出最新這一句的翻譯結果，不要翻譯整段上下文。\n"
                        "56. 如果上下文不足，請用家庭照護生活情境中最自然、最可能的意思翻譯。\n"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "以下是最近對話上下文，僅供判斷稱呼、人物與語氣，不要翻譯整段上下文：\n"
                        f"{context_text}\n\n"
                        "請根據上述家庭照護情境與最近上下文，只翻譯最新這一句。\n"
                        "若印尼文文法不完整、拼字不標準或沒有標點，請推測最自然的生活語意，不要逐字硬翻。\n"
                        "最終只輸出翻譯結果。\n\n"
                        "最新句子：\n"
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
