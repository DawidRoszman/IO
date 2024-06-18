import praw
import pandas as pd

# Set up your Reddit app credentials
reddit = praw.Reddit(
    client_id="WsZKJ3Y5X2GGGBsYI70IHQ",
    client_secret="dun-qp-cbScmKhrAQz0DynjyYlXpjQ",
    user_agent="io-lab by u/Accomplished_Mix2408",
)

# Define the subreddit
subreddit = reddit.subreddit("cybersecurity")

# Scrape 100 posts from the subreddit
posts = []
for post in subreddit.new(limit=100):
    posts.append(
        {
            "date": post.created_utc,
            "title": post.title,
            "content": post.selftext,
            "score": post.score,
            "url": post.url,
            "num_comments": post.num_comments,
        }
    )

# Convert the list of posts to a DataFrame
df = pd.DataFrame(posts)

# Save the DataFrame to a CSV file
file_path = "cybersecurity_posts.csv"
df.to_csv(file_path, index=False)

print(f"Scraped posts saved to {file_path}")
