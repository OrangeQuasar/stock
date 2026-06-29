import feedparser

RSS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://finance.yahoo.com/news/rssindex"
]

def fetch_news():

    text = ""

    for url in RSS:

        feed = feedparser.parse(url)

        for item in feed.entries[:5]:

            text += item.title + "\n"

    return text