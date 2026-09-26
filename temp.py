from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv
from openai import OpenAI
import os
import logging
import re

# ============================================================
# 基本設定
# ============================================================

load_dotenv()

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 翻譯準確度優先
# Render 可另外設定 OPENAI_MODEL 覆蓋
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")

required_env = {
    "LINE_CHANNEL_ACCESS_TOKEN": LINE_CHANNEL_ACCESS_TOKEN,
    "LINE_CHANNEL_SECRET": LINE_CHANNEL_SECRET,
    "OPENAI_API_KEY": OPENAI_API_KEY,
}

missing = [key for key, value in required_env.items() if not value]

if missing:
    raise RuntimeError(
        f"缺少必要環境變數：{', '.join(missing)}"
    )

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)


# ============================================================
# 語言判斷
# ============================================================

def contains_chinese(text: str) -> bool:
    """
    判斷文字是否含有中文漢字。
    """
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def chinese_ratio(text: str) -> float:
    """
    計算中文字在有效文字中的比例。
    用來判斷翻成印尼文後是否仍然錯誤輸出大量中文。
    """

    chars = [
        c for c in text
        if not c.isspace() and not re.match(r'[\W_]', c)
    ]

    if not chars:
        return 0.0

    chinese_count = sum(
        1 for c in chars
        if '\u4e00' <= c <= '\u9fff'
    )

    return chinese_count / len(chars)


# ============================================================
# OpenAI 翻譯
# ============================================================

def call_translation_model(
    user_text: str,
    direction: str,
    retry: bool = False
) -> str:

    if direction == "zh_to_id":

        instructions = """
你是中文到印尼文的專業生活對話翻譯器。

使用情境：
台灣家庭成員與印尼籍家庭看護之間的 LINE 日常對話。

你的任務只有一個：
把使用者輸入的繁體中文，翻譯成自然、清楚、容易讓印尼籍家庭看護理解的印尼文。

翻譯原則：

- 先理解整句真正想表達的意思，再翻譯。
- 不要逐字翻譯中文語序。
- 要使用印尼人日常聊天會使用的自然說法。
- 語氣可以口語，但意思必須完整清楚。
- 特別注意因果、時間、否定、已經、還沒、正在、等一下、以後等語意。
- 不要自行加入原文沒有的要求、責備或情緒。
- 如果中文有省略主詞，可以按照生活語境自然翻譯。
- 「奶奶」通常翻譯為 Nenek。
- 「媽媽」通常翻譯為 Mama。
- 「哥哥」翻譯為 Kakak 或 Kak，以自然語境為準。
- 「姐姐」需要明確表達女性時，可使用 kakak perempuan。
- 「吃藥」請使用自然印尼文 minum obat。
- 「拍影片／錄影片」使用 buat video、rekam video 或自然等價表達。
- 「出門」不要錯翻成 jalan keluar（出口），應依語境使用 keluar / pergi keluar 等自然說法。

最重要：

最終答案必須只有印尼文。
禁止輸出中文。
禁止解釋。
禁止寫「翻譯如下」。
禁止加引號。
禁止回覆原文。

即使中文句子很口語、沒有標點，也要先理解意思再翻成完整自然的印尼文。
"""

        if retry:
            instructions += """

你上一次的輸出沒有完全符合要求。
這次請特別確認：
輸出只能有印尼文，不可以包含中文漢字。
"""

    else:

        instructions = """
你是印尼文到繁體中文的專業生活對話翻譯器。

使用情境：
印尼籍家庭看護與台灣家庭成員之間的 LINE 日常對話。

印尼籍看護的文字可能：
- 沒有標點
- 拼字錯誤
- 文法不完整
- 省略主詞
- 省略因為、所以、已經、還沒等詞
- 使用非常口語的印尼文

你的任務不是逐字翻譯，而是先理解她真正想表達的生活語意，再翻成自然的台灣繁體中文。

翻譯原則：

- 使用繁體中文。
- 使用台灣人自然的口語。
- 保留原句真正的語氣，例如回報、詢問、說明、提醒。
- 原文不是問句時，不要擅自改成問句。
- 原文是問句時，要翻成自然問句。
- 不要自行加入責備、情緒或原文沒有的資訊。

家庭稱呼：

- Nenek / nenek → 奶奶
- Mama / mama → 媽媽
- kak / kakak 預設 → 哥哥
- 只有非常明確是女性時才翻成姐姐
- tuan 視情況翻成先生，也可以自然省略

常見生活語意：

- belum → 還沒
- sudah → 已經
- sedang / lagi → 正在
- mau → 要／想要／準備要
- nanti → 等一下／晚一點
- sekarang → 現在
- makan → 吃／吃飯
- makan siang → 吃午餐／午餐
- tidur → 睡覺
- mandi → 洗澡
- minum obat → 吃藥
- bikin video / bikin vidio / buat video → 拍影片／錄影片

請特別理解不標準口語。

例如：

Ada kakak tidak bikin vidio makan siang

比較自然的意思是：

有哥哥在，所以沒有拍午餐影片。

不是：

有哥哥沒有拍影片嗎？

又例如：

Nenek tidur tuan apakah jalan keluar tuan

自然理解為：

奶奶睡了，先生要出門嗎？

不要把 jalan keluar 固定理解為「出口」。

最重要：

最終答案只能輸出繁體中文翻譯。
不要說明。
不要分析。
不要寫「翻譯如下」。
不要加引號。
不要把印尼文原文一起輸出。
"""

        if retry:
            instructions += """

你上一次沒有正確輸出繁體中文。
這次請只輸出繁體中文翻譯。
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=user_text
    )

    result = response.output_text

    if not result:
        raise RuntimeError("OpenAI 沒有回傳翻譯內容")

    return result.strip()


# ============================================================
# 主翻譯邏輯
# ============================================================

def translate_text(text: str) -> str:

    if not text or not text.strip():
        return ""

    user_text = text.strip()

    try:

        # ----------------------------------------------------
        # 中文 → 印尼文
        # ----------------------------------------------------
        if contains_chinese(user_text):

            translated = call_translation_model(
                user_text,
                "zh_to_id"
            )

            # 防呆：
            # 如果翻譯結果仍然出現明顯大量中文，
            # 自動再要求模型翻一次
            if chinese_ratio(translated) > 0.05:

                logging.warning(
                    "中文→印尼文結果仍含中文，自動重試：%s",
                    translated
                )

                translated = call_translation_model(
                    user_text,
                    "zh_to_id",
                    retry=True
                )

            return translated

        # ----------------------------------------------------
        # 印尼文 → 繁體中文
        # ----------------------------------------------------
        else:

            translated = call_translation_model(
                user_text,
                "id_to_zh"
            )

            # 印尼文翻中文正常情況應該會包含中文字
            # 若完全沒有中文字，自動再試一次
            if not contains_chinese(translated):

                logging.warning(
                    "印尼文→中文結果未包含中文，自動重試：%s",
                    translated
                )

                translated = call_translation_model(
                    user_text,
                    "id_to_zh",
                    retry=True
                )

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


# ============================================================
# LINE 訊息處理
# ============================================================

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):

    user_message = event.message.text

    translated = translate_text(user_message)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=translated)
    )


# ============================================================
# 本機測試
# Render 通常使用 gunicorn app:app
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
