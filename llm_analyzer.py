import ollama
from config import OLLAMA_MODEL

def analyze_news(news):

    prompt = f"""
次のニュースが翌日の日本株市場に
与える影響を評価してください。

ニュース:
{news}

-5から+5の整数のみ返してください。
"""

    try:

        r = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )

        score = int(
            ''.join(
                filter(
                    lambda x:x in "-0123456789",
                    r["message"]["content"]
                )
            )
        )

        return score

    except:
        return 0