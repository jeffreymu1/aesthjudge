import pandas as pd
import requests
import os

### OUTPUT HERE
OUTPUT_FILE = "scraped.csv"

# save the progress of the data collection

def save_progress(data):
    pd.DataFrame(data).to_csv(OUTPUT_FILE, index=False)


def load_progress():
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        collected_data = df.to_dict(orient="records")
        processed_urls = set(df['url'])

    else:
        collected_data = []
        processed_urls = set()

    return collected_data, processed_urls


def fetch_pushshift(subreddit, before, size):
    url = "https://api.pushshift.io/reddit/submission/search/"
    params = {
        "subreddit": subreddit,
        "size": size,
        "before": before,
        "sort": "desc",
        "is_video": False,
        "filter": ["id", "url", "title", "score", "num_comments", "created_utc"]
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print(f"pushshift error: {e}")
        return []