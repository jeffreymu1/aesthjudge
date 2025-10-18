from img_prop_helpers import *
from img_content_helpers import *
from sentiment_analysis import sentiment_score
from meta_fns import save_progress, load_progress

from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from tqdm import tqdm
import praw
import numpy as np
import cv2
import os
import requests

load_dotenv()

client_id = os.getenv('RD_CLIENT_ID')
client_secret = os.getenv('RD_CLIENT_SECRET')
user_agent = os.getenv('RD_USER_AGENT')

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# subreddit here
subreddit = reddit.subreddit("")

# collect here
collected_data = []

# unload
collected_data, processed_urls = load_progress()

for sub in tqdm(subreddit.top(time_filter='all', limit=None)):
    try:
        # if processed skip
        if sub.url in processed_urls:
            continue

        # skip posts without comments
        if sub.num_comments == 0:
            continue

        # download image
        response = requests.get(sub.url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # all top-level comments
        sub.comments.replace_more(limit=0)
        top_comments = [c.body for c in sub.comments.list() if c.parent_id == c.link_id]
        comments_joined = " ".join(top_comments).strip()

        if not comments_joined:
            continue

        # compute img features
        subj_type, subj_conf = classify_subject_type(img_cv)
        genre, genre_conf = classify_genre(img_cv)
        perspective, pers_conf = classify_perspective(img_cv)
        color_mode, color_conf = classify_color_mode(img_cv)
        complexity, comp_conf = classify_complexity(img_cv)

        # compute target var
        scores = [sentiment_score(c) for c in top_comments if len(c.strip()) > 0]
        aesthetic_score = np.mean(scores)

        features = {
            # post metadata
            "subreddit": sub.subreddit.display_name,
            "title": sub.title,
            "upvotes": sub.score,
            "upvote_ratio": sub.upvote_ratio,
            "num_comments": sub.num_comments,
            "url": sub.url,
            "vote-to-comment": sub.score/sub.num_comments if sub.num_comments > 0 else 0,
            
            # visual features
            "brightness": brightness(img_cv),
            "contrast": contrast(img_cv),
            "saturation": saturation(img_cv),
            "symmetry_lr": symmetry_lr(img_cv),
            "symmetry_td": symmetry_td(img_cv),
            "light_balance_lr": light_balance_lr(img_cv),
            "light_balance_td": light_balance_td(img_cv),
            "hue_diversity": hue_diversity(img_cv),
            "dynamic_range": dynamic_range(img_cv),
            "sharpness": sharpness(img_cv),
            "thirds_balance": thirds_balance(img_cv),
            "texture_variance": texture_variance(img_cv),

            # image contents
            "subject_type": subj_type,
            "genre": genre,
            "perspective": perspective,
            "color_mode": color_mode,
            "complexity": complexity,
            
            # target variable
            "mean_sentiment": aesthetic_score

        }
        collected_data.append(features)
        processed_urls.add(sub.url)

        if len(collected_data) % 10 == 0:
            save_progress(collected_data)

    except KeyboardInterrupt:
        print("exiting early, saving progress...")
        save_progress(collected_data)
        break

    except Exception as e:
        print(f"SKIPPING b/c error w/ {sub.url}: {e}\n")


# final save
save_progress(collected_data)
print(f"total posts collected: {len(collected_data)}")


'''
    if method_name == 'top':
    elif method_name == 'hot':
    elif method_name == 'new':
    elif method_name == 'controversial':
    elif method_name == 'rising':

'day', 'year', 'month', 'hour', 'week', 'all'

'''