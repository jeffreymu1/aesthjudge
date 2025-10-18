import pandas as pd
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