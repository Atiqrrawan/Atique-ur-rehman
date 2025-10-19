import os
import requests
from datetime import datetime, timedelta
import streamlit as st

API_KEY = os.getenv("AIzaSyByzQ8lLCore68ZNcm3tjyB0QDYKobLXEA")

# API URLs
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"

# -----------------------------
# 📦 Utility Functions
# -----------------------------

def fetch_videos_for_keyword(keyword, published_after, max_results=5):
    params = {
        "part": "snippet",
        "q": keyword,
        "type": "video",
        "order": "viewCount",
        "publishedAfter": published_after,
        "maxResults": max_results,
        "relevanceLanguage": "en",
        "regionCode": "US",
        "key": API_KEY
    }
    response = requests.get(SEARCH_URL, params=params)
    response.raise_for_status()
    return response.json().get("items", [])

def fetch_video_stats(video_ids):
    params = {
        "part": "snippet,statistics,contentDetails",
        "id": ",".join(video_ids),
        "key": API_KEY
    }
    response = requests.get(VIDEOS_URL, params=params)
    response.raise_for_status()
    return response.json().get("items", [])

def fetch_channel_stats(channel_ids):
    params = {
        "part": "snippet,statistics",
        "id": ",".join(channel_ids),
        "key": API_KEY
    }
    response = requests.get(CHANNELS_URL, params=params)
    response.raise_for_status()
    return response.json().get("items", [])

# -----------------------------
# 🌐 Streamlit UI
# -----------------------------

st.set_page_config(page_title="YouTube History Channel Analyzer", layout="wide")
st.title("📺 YouTube Viral History Channel Analyzer")

days = st.number_input("🔎 Enter Days to Search (1–30):", min_value=1, max_value=30, value=7)
min_subscribers = st.number_input("👤 Max Subscribers (to filter smaller channels):", min_value=0, value=3000)

# Search Keywords
keywords = [
    "unsolved historical cases", "lost civilizations", "secret societies history",
    "hidden treasures", "unexplained events in history", "royal family secrets",
    "vanished empires", "dark history", "historical conspiracy theories",
    "rare historical facts", "mysterious deaths in history", "war mysteries"
]

if st.button("🚀 Fetch Viral Data"):
    start_date = (datetime.utcnow() - timedelta(days=int(days))).isoformat("T") + "Z"
    all_results = []

    for keyword in keywords:
        st.write(f"🔍 Searching for keyword: **{keyword}** …")
        try:
            videos = fetch_videos_for_keyword(keyword, start_date, max_results=5)
        except Exception as e:
            st.warning(f"⚠️ API error for '{keyword}': {e}")
            continue

        if not videos:
            st.info(f"ℹ️ No videos found for keyword: {keyword}")
            continue

        video_ids = [v["id"]["videoId"] for v in videos if "videoId" in v["id"]]
        channel_ids = [v["snippet"]["channelId"] for v in videos]

        try:
            video_stats = fetch_video_stats(video_ids)
            channel_stats = fetch_channel_stats(channel_ids)
        except Exception as e:
            st.warning(f"⚠️ Error fetching stats for '{keyword}': {e}")
            continue

        for vid, chan in zip(video_stats, channel_stats):
            try:
                channel_id = chan["id"]
                channel_title = chan["snippet"].get("title", "N/A")
                channel_desc = chan["snippet"].get("description", "")[:300]
                subs = int(chan["statistics"].get("subscriberCount", 0))
                channel_views = int(chan["statistics"].get("viewCount", 0))

                video_title = vid["snippet"].get("title", "N/A")
                video_desc = vid["snippet"].get("description", "")[:300]
                video_url = f"https://www.youtube.com/watch?v={vid['id']}"
                views = int(vid["statistics"].get("viewCount", 0))
                likes = int(vid["statistics"].get("likeCount", 0))
                comments = int(vid["statistics"].get("commentCount", 0))

                if subs <= min_subscribers:
                    all_results.append({
                        "Channel ID": channel_id,
                        "Channel Title": channel_title,
                        "Channel Description": channel_desc,
                        "Subscribers": subs,
                        "Channel Views": channel_views,
                        "Video Title": video_title,
                        "Video URL": video_url,
                        "Video Views": views,
                        "Likes": likes,
                        "Comments": comments,
                        "Keyword": keyword
                    })
            except Exception as inner_e:
                st.warning(f"❌ Error processing a video: {inner_e}")
                continue

    # -----------------------------
    # 🧾 Display Results
    # -----------------------------
    if all_results:
        st.success(f"✅ Found {len(all_results)} videos from smaller channels.")
        for r in all_results:
            st.markdown(
                f"""
                ### 🎬 {r['Video Title']}
                **Channel:** {r['Channel Title']}  
                **Subscribers:** {r['Subscribers']:,}  
                **Channel Views:** {r['Channel Views']:,}  
                **Video Views:** {r['Video Views']:,}  
                **Likes:** {r['Likes']:,} | **Comments:** {r['Comments']:,}  
                **Keyword Used:** *{r['Keyword']}*  
                🔗 [Watch on YouTube]({r['Video URL']})
                ---
                """
            )
    else:
        st.warning("😕 No matching videos found for smaller channels.")
