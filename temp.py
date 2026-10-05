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

missing = [
    key for key, value in required_env.items()
    if not value
]

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
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def chinese_ratio(text: str) -> float:
    """
    計算中文字在有效文字中的比例。

    用來判斷中文翻成印尼文之後，
    是否仍然錯誤輸出大量中文。
    """

    chars = [
        c for c in text
        if not c.isspace()
        and not re.match(r"[\W_]", c)
    ]

    if not chars:
        return 0.0

    chinese_count = sum(
        1 for c in chars
        if "\u4e00" <= c <= "\u9fff"
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

    # --------------------------------------------------------
    # 中文 → 印尼文
    # --------------------------------------------------------
    if direction == "zh_to_id":

        instructions = """
你是中文到印尼文的專業生活對話翻譯器。

使用情境：
台灣家庭成員與印尼籍家庭看護之間的 LINE 日常對話。

你的任務只有一個：
把使用者輸入的繁體中文，
翻譯成自然、清楚、容易讓印尼籍家庭看護理解的印尼文。

【翻譯原則】

- 先理解整句真正想表達的意思，再翻譯。
- 不要逐字翻譯中文語序。
- 使用印尼人日常聊天會使用的自然說法。
- 語氣可以口語，但意思必須完整清楚。
- 特別注意因果、時間、否定、已經、還沒、正在、
  等一下、以後等語意。
- 不要自行加入原文沒有的要求、責備或情緒。
- 如果中文有省略主詞，可以按照生活語境自然翻譯。

【家庭稱呼】

- 「奶奶」通常翻譯為 Nenek。
- 「媽媽」通常翻譯為 Mama。
- 「哥哥」翻譯為 Kakak 或 Kak，以自然語境為準。
- 「姐姐」需要明確表達女性時，可使用 kakak perempuan。

【常見生活用語】

- 「吃藥」使用自然印尼文 minum obat。
- 「拍影片／錄影片」使用
  buat video、rekam video 或自然等價表達。
- 「出門」不要錯翻成 jalan keluar（出口），
  應依語境使用 keluar、pergi keluar 等自然說法。

【重要】

準確度高於逐字對應。

即使中文句子很口語、沒有標點，
也要先理解意思，
再翻成完整、自然、容易理解的印尼文。

【輸出規則】

最終答案必須只有印尼文。
禁止輸出中文。
禁止解釋。
禁止寫「翻譯如下」。
禁止加引號。
禁止回覆原文。
"""

        if retry:
            instructions += """

你上一次的輸出沒有完全符合要求。

這次請特別確認：
輸出只能有印尼文，
不可以包含中文漢字。
"""

    # --------------------------------------------------------
    # 印尼文 → 繁體中文
    # --------------------------------------------------------
    elif direction == "id_to_zh":

        instructions = """
你是印尼文到繁體中文的專業生活對話翻譯器。

使用情境：
印尼籍家庭看護與台灣家庭成員之間的 LINE 日常對話。

【重要前提】

你只能看到目前這一則訊息。

不要假設有前文。
不要自行建立不存在的上下文。
不要根據想像補充對方沒有說的事情。

印尼籍看護輸入的文字可能：

- 沒有標點
- 拼字錯誤
- 文法不完整
- 省略主詞
- 省略部分連接詞
- 使用非常口語的印尼文
- 使用非標準書面用法
- 使用意思相近但不完全精確的詞

你的任務：

先理解目前這一句最合理的生活語意，
再翻成自然的台灣繁體中文。

但是：

「理解語意」不代表可以替說話者猜測
原文沒有明確表達的意思。

============================================================
準確度優先原則
============================================================

1.
不要只做機械式逐字翻譯，
但也不要過度推測。

2.
可以根據目前整句的：

- 文法
- 詞義
- 前後因果
- 否定
- 時間
- 語氣

修正明顯的口語、省略或小型拼字錯誤。

3.
如果原文某個詞或片語存在兩種以上合理解釋，
而目前這一句沒有足夠資訊判斷，

不可以自行選擇其中一個更具體的意思。

4.
遇到歧義時：

優先使用可以保留原文模糊程度的中文。

寧可稍微保守，
也不要加入原文沒有明確表達的資訊。

5.
不要為了讓中文讀起來更漂亮，
把模糊的原文翻譯成非常具體的行為。

============================================================
非常重要的歧義處理
============================================================

例如：

mengembangkan gaji saya

這種說法本身不是非常標準，
可能存在不同理解。

如果目前只有這一句，
不要擅自翻譯成：

「幫我調薪」

因為這會加入「雇主提高薪資」的具體意思。

也不要擅自翻譯成：

「幫我投資薪水」

因為這會加入「投資」的具體意思。

如果原文沒有足夠資訊，
應保留原文比較模糊的意思，例如：

「幫我讓我的薪水增值」

或其他不增加額外意思的自然中文。

只有當原文明確使用：

menaikkan gaji

或整句明確是在談薪資調整時，
才翻譯成：

「加薪」
「調薪」
「提高薪水」

只有當原文明確使用：

investasi
menginvestasikan
berinvestasi

或整句非常明確是在談投資時，
才可以翻譯成：

「投資」
「拿去投資」

============================================================
翻譯原則
============================================================

- 使用繁體中文。
- 使用台灣人自然的口語。
- 保留原句真正的語氣。
- 原文不是問句時，不要擅自改成問句。
- 原文是問句時，要翻成自然問句。
- 不要自行加入責備。
- 不要自行加入情緒。
- 不要自行加入人物關係。
- 不要自行加入原文沒有的資訊。

如果無法確定：

保留原文的模糊程度，
不要猜得比原文更具體。

============================================================
家庭稱呼
============================================================

- Nenek / nenek → 奶奶
- Mama / mama → 媽媽
- kak / kakak 預設 → 哥哥
- 只有非常明確是女性時才翻成姐姐
- tuan 視情況翻成先生，也可以自然省略

============================================================
常見生活語意
============================================================

- belum → 還沒
- sudah → 已經
- sedang / lagi → 正在
- mau → 要／想要／準備要
- nanti → 等一下／晚一點／之後
- sekarang → 現在
- makan → 吃／吃飯
- makan siang → 吃午餐／午餐
- tidur → 睡覺
- mandi → 洗澡
- minum obat → 吃藥
- bikin video / bikin vidio / buat video
  → 拍影片／錄影片

============================================================
非標準口語範例
============================================================

例如：

Ada kakak tidak bikin vidio makan siang

比較自然的理解是：

有哥哥在，所以沒有拍午餐影片。

不要翻譯成：

有哥哥沒有拍影片嗎？

------------------------------------------------------------

例如：

Nenek tidur tuan apakah jalan keluar tuan

自然理解為：

奶奶睡了，先生要出門嗎？

不要把：

jalan keluar

固定理解成：

出口

============================================================
核心規則
============================================================

準確度高於中文的漂亮程度。

可以修正文法，
可以理解口語，
可以理解小型拼字錯誤，

但是：

不可以把「可能是這個意思」
翻譯成
「確定就是這個意思」。

如果資訊不足，
請保留原文的模糊程度。

============================================================
輸出規則
============================================================

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
不要解釋。
不要輸出印尼文原文。
"""

    else:
        raise ValueError(
            f"不支援的翻譯方向：{direction}"
        )

    # --------------------------------------------------------
    # 呼叫 OpenAI Responses API
    # --------------------------------------------------------

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=user_text
    )

    result = response.output_text

    if not result:
        raise RuntimeError(
            "OpenAI 沒有回傳翻譯內容"
        )

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
                user_text=user_text,
                direction="zh_to_id"
            )

            # 防呆：
            # 如果翻譯結果仍然出現明顯大量中文，
            # 自動再要求模型翻一次。

            if chinese_ratio(translated) > 0.05:

                logging.warning(
                    "中文→印尼文結果仍含中文，自動重試：%s",
                    translated
                )

                translated = call_translation_model(
                    user_text=user_text,
                    direction="zh_to_id",
                    retry=True
                )

            return translated

        # ----------------------------------------------------
        # 印尼文 → 繁體中文
        # ----------------------------------------------------

        else:

            translated = call_translation_model(
                user_text=user_text,
                direction="id_to_zh"
            )

            # 印尼文翻中文正常應包含中文字。
            # 如果完全沒有中文字，自動重試。

            if not contains_chinese(translated):

                logging.warning(
                    "印尼文→中文結果未包含中文，自動重試：%s",
                    translated
                )

                translated = call_translation_model(
                    user_text=user_text,
                    direction="id_to_zh",
                    retry=True
                )

            return translated

    except Exception as e:

        logging.exception(
            "OpenAI 翻譯失敗"
        )

        return f"翻譯失敗：{str(e)}"


# ============================================================
# Render / LINE Webhook
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return "LINE Bot 已上線"


@app.route("/callback", methods=["POST"])
def callback():

    signature = request.headers.get(
        "X-Line-Signature"
    )

    body = request.get_data(
        as_text=True
    )

    if not signature:
        abort(400)

    try:

        handler.handle(
            body,
            signature
        )

    except InvalidSignatureError:

        abort(400)

    except Exception:

        logging.exception(
            "LINE callback 發生錯誤"
        )

        abort(500)

    return "OK"


# ============================================================
# LINE 訊息處理
# ============================================================

@handler.add(
    MessageEvent,
    message=TextMessage
)
def handle_message(event):

    user_message = event.message.text

    translated = translate_text(
        user_message
    )

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text=translated
        )
    )


# ============================================================
# 本機測試
#
# Render 正式環境通常使用：
#
# gunicorn app:app
#
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
