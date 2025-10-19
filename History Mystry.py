import os
import requests
from datetime import datetime, timedelta
import streamlit as st

API_KEY = os.getenv("AIzaSyAGCbuynm_B6yr16ISp0Igt_DNDpUgGFVw")
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"

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
    resp = requests.get(SEARCH_URL, params=params)
    resp.raise_for_status()
    return resp.json().get("items", [])

def fetch_video_stats(video_ids):
    params = {
        "part": "snippet,statistics,contentDetails",
        "id": ",".join(video_ids),
        "key": API_KEY
    }
    resp = requests.get(VIDEOS_URL, params=params)
    resp.raise_for_status()
    return resp.json().get("items", [])

def fetch_channel_stats(channel_ids):
    params = {
        "part": "snippet,statistics",
        "id": ",".join(channel_ids),
        "key": API_KEY
    }
    resp = requests.get(CHANNELS_URL, params=params)
    resp.raise_for_status()
    return resp.json().get("items", [])

# Streamlit UI
st.title("YouTube History‑Channel Growth & Viral Video Analyzer")

days = st.number_input("Enter Days to Search (1‑30):", min_value=1, max_value=30, value=7)
min_subscribers = st.number_input("Maximum Subscribers of Channel (to find smaller channels):", min_value=0, value=3000)
if st.button("Fetch Data"):
    start_date = (datetime.utcnow() - timedelta(days=int(days))).isoformat("T") + "Z"
    all_results = []
    keywords = [
        "unsolved historical cases", "lost civilizations", "secret societies history",
        "hidden treasures", "unexplained events in history", "royal family secrets",
        "vanished empires", "dark history", "historical conspiracy theories",
        "rare historical facts", "mysterious deaths in history", "war mysteries"
    ]
    for kw in keywords:
        st.write(f"Searching for keyword: **{kw}** …")
        try:
            videos = fetch_videos_for_keyword(kw, start_date, max_results=5)
        except Exception as e:
            st.warning(f"API error for keyword '{kw}': {e}")
            continue

        if not videos:
            st.info(f"No videos found for keyword: {kw}")
            continue

        video_ids = [v["id"]["videoId"] for v in videos if "videoId" in v["id"]]
        channel_ids = [v["snippet"]["channelId"] for v in videos]

        if not video_ids or not channel_ids:
            st.warning(f"Skipping keyword '{kw}' due to missing ids.")
            continue

        try:
            video_stats = fetch_video_stats(video_ids)
            channel_stats = fetch_channel_stats(channel_ids)
        except Exception as e:
            st.warning(f"Error fetching stats for keyword '{kw}': {e}")
            continue

        for vid_item, chan_item in zip(video_stats, channel_stats):
            channel_id = chan_item["id"]
            title = vid_item["snippet"].get("title", "N/A")
            description = vid_item["snippet"].get("description", "")[:300]
            video_url = f"https://www.youtube.com/watch?v={vid_item['id']}"
            viewCount = int(vid_item["statistics"].get("viewCount", 0))
            likeCount = int(vid_item["statistics"].get("likeCount", 0))
            commentCount = int(vid_item["statistics"].get("commentCount", 0))
            subscriberCount = int(chan_item["statistics"].get("subscriberCount", 0))
            channelViewCount = int(chan_item["statistics"].get("viewCount", 0))
            channelTitle = chan_item["snippet"].get("title", "N/A")
            channelDescription = chan_item["snippet"].get("description", "")[:300]

                      if subscriberCount <= min_subscribers:
                all_results.append({
                    "Channel ID": channel_id,
                    "Channel Title": channelTitle,
                    "Channel Description": channelDescription,
                    "Subscriber Count": subscriberCount,
                    "Channel View Count": channelViewCount,
                    "Video Title": title,
                    "Video URL": video_url,
                    "Video Views": viewCount,
                    "Video Likes": likeCount,
                    "Video Comments": commentCount,
                    "Keyword": kw
                })

    if all_results:
        st.success(f"Found {len(all_results)} video results from smaller channels.")
        for r in all_results:
            st.markdown(
                f"### Channel: {r['Channel Title']}  \n"
                f"**Channel ID:** {r['Channel ID']}  \n"
                f"**Subscribers:** {r['Subscriber Count']}  \n"
                f"**Channel Total Views:** {r['Channel View Count']}  \n"
                f"**Video Title:** {r['Video Title']}  \n"
                f"**Video URL:** [Watch]({r['Video URL']})  \n"
                f"**Video Views:** {r['Video Views']}  \n"
                f"**Likes:** {r['Video Likes']}  \n"
                f"**Comments:** {r['Video Comments']}  \n"
                f"**Keyword Searched:** {r['Keyword']}  \n"
                f"---"
            )
    else:
        st.warning("No eligible results found (with subscribers ≤ threshold).")
