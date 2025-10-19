import requests
from datetime import datetime, timedelta
import streamlit as st

API_KEY = "AIzaSyByzQ8lLCore68ZNcm3tjyB0QDYKobLXEA"

# YouTube API URLs
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"

# ------------------------------------
# 📦 Functions
# ------------------------------------
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
        "part": "snippet,statistics",
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

# ------------------------------------
# 🌐 Streamlit App
# ------------------------------------
st.set_page_config(page_title="YouTube History Analyzer", layout="wide")
st.title("📺 Viral History YouTube Channel Finder")

# Input
days = st.number_input("Search videos from past N days:", min_value=1, max_value=30, value=7)
max_subs = st.number_input("Only show channels with subscribers less than:", min_value=0, value=3000)

# Keywords
keywords = [
    "unsolved historical cases", "lost civilizations", "secret societies history",
    "hidden treasures", "unexplained events in history", "royal family secrets",
    "vanished empires", "dark history", "historical conspiracy theories",
    "rare historical facts", "mysterious deaths in history", "war mysteries"
]

if st.button("🔍 Fetch YouTube Data"):
    published_after = (datetime.utcnow() - timedelta(days=days)).isoformat("T") + "Z"
    all_results = []

    for keyword in keywords:
        st.write(f"🔎 Searching: **{keyword}**")
        try:
            videos = fetch_videos_for_keyword(keyword, published_after, max_results=5)
            if not videos:
                st.info(f"No videos found for: {keyword}")
                continue

            video_ids = [v["id"]["videoId"] for v in videos if "videoId" in v["id"]]
            channel_ids = [v["snippet"]["channelId"] for v in videos]

            video_stats = fetch_video_stats(video_ids)
            channel_stats = fetch_channel_stats(channel_ids)

            for vid, chan in zip(video_stats, channel_stats):
                try:
                    subs = int(chan["statistics"].get("subscriberCount", 0))
                    if subs > max_subs:
                        continue

                    video_data = {
                        "Video Title": vid["snippet"].get("title", "N/A"),
                        "Video URL": f"https://www.youtube.com/watch?v={vid['id']}",
                        "Video Views": int(vid["statistics"].get("viewCount", 0)),
                        "Likes": int(vid["statistics"].get("likeCount", 0)),
                        "Comments": int(vid["statistics"].get("commentCount", 0)),
                        "Channel Title": chan["snippet"].get("title", "N/A"),
                        "Subscribers": subs,
                        "Channel Views": int(chan["statistics"].get("viewCount", 0)),
                        "Keyword": keyword
                    }
                    all_results.append(video_data)
                except Exception as e:
                    st.warning(f"⚠️ Error parsing data: {e}")
                    continue

        except Exception as e:
            st.error(f"❌ API error: {e}")

    # ------------------------------------
    # ✅ Display Results
    # ------------------------------------
    if all_results:
        st.success(f"Found {len(all_results)} videos from small channels.")
        for item in all_results:
            st.markdown(
                f"""
                ### 🎬 {item['Video Title']}
                **Channel:** {item['Channel Title']}  
                **Subscribers:** {item['Subscribers']:,}  
                **Channel Views:** {item['Channel Views']:,}  
                **Video Views:** {item['Video Views']:,}  
                **Likes:** {item['Likes']:,} | Comments: {item['Comments']:,}  
                **Keyword:** *{item['Keyword']}*  
                🔗 [Watch Video]({item['Video URL']})
                ---
                """
            )
    else:
        st.warning("No results found.")

