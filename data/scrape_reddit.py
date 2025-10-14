import praw
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

client_id = os.getenv('RD_CLIENT_ID')
client_secret = os.getenv('RD_CLIENT_SECRET')
user_agent = os.getenv('RD_USER_AGENT')

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

subreddit = reddit.subreddit("itookapicture")  

