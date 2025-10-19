import streamlit as st
import requests
from datetime import datetime, timedelta

# YouTube API Key
API_KEY = "AIzaSyAGCbuynm_B6yr16ISp0Igt_DNDpUgGFVw"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"

# Streamlit App Title
st.title("📽️ Viral YouTube History Topics (US, UK, Canada)")

# Input Fields
days = st.number_input("Enter Days to Search (1-30):", min_value=1, max_value=30, value=5)

# Cleaned, deduplicated, relevant keywords
keywords = list(set([
    "unsolved historical cases", "Reddit Update", "Reddit lost civilizations", 
    "secret societies history", "forbidden history", "archaeological discoveries", 
    "missing persons history", "hidden treasures", "unexplained events in history", 
    "royal family secrets", "vanished empires", "dark history", "historical conspiracy theories", 
    "rare historical facts", "mysterious deaths in history", "war mysteries", 
    "historical anomalies", "famous heists in history", "hidden languages and scripts", 
    "British royal scandals", "Ancient Rome mysteries", "American civil war secrets", 
    "Titanic conspiracy", "Victorian era dark secrets", "Reddit infidelity confessions", 
    "Reddit relationship drama", "Canadian war history", "UK lost castles", "colonial America secrets", 
    "Reddit paranormal history", "creepy historical events", "bizarre history facts", 
    "YouTube dark history", "mystery of the pyramids", "Illuminati in history"
]))

# Fetch Data Button
if st.button("🔍 Fetch Viral History Videos"):
    try:
        start_date = (datetime.utcnow() - timedelta(days=int(days))).isoformat("T") + "Z"
        all_results = []

        for keyword in keywords:
            st.write(f"🔎 Searching for: **{keyword}**")

            search_params = {
                "part": "snippet",
                "q": keyword,
                "type": "video",
                "order": "viewCount",
                "publishedAfter": start_date,
                "maxResults": 5,
                "regionCode": "US",  # Optional: prioritize US/Canada/UK
                "relevanceLanguage": "en",
                "key": API_KEY,
            }

            response = requests.get(YOUTUBE_SEARCH_URL, params=search_params)
            if response.status_code != 200:
                st.warning(f"⚠️ API error for keyword '{keyword}': {response.status_code}")
                continue

            data = response.json()

            if "items" not in data or not data["items"]:
                st.info(f"No videos found for: {keyword}")
                continue

            videos = data["items"]
            video_ids = [v["id"]["videoId"] for v in videos if "id" in v and "videoId" in v["id"]]
            channel_ids = [v["snippet"]["channelId"] for v in videos if "snippet" in v and "channelId" in v["snippet"]]

            if not video_ids or not channel_ids:
                st.warning(f"Skipping '{keyword}' due to missing video/channel data.")
                continue

            # Get video stats
            stats_params = {"part": "statistics", "id": ",".join(video_ids), "key": API_KEY}
            stats_response = requests.get(YOUTUBE_VIDEO_URL, params=stats_params)
            stats_data = stats_response.json()

            if "items" not in stats_data or not stats_data["items"]:
                st.warning(f"Failed to fetch video stats for '{keyword}'")
                continue

            # Get channel stats
            channel_params = {"part": "statistics", "id": ",".join(channel_ids), "key": API_KEY}
            channel_response = requests.get(YOUTUBE_CHANNEL_URL, params=channel_params)
            channel_data = channel_response.json()

            if "items" not in channel_data or not channel_data["items"]:
                st.warning(f"Failed to fetch channel stats for '{keyword}'")
                continue

            for video, stat, channel in zip(videos, stats_data["items"], channel_data["items"]):
                title = video["snippet"].get("title", "N/A")
                description = video["snippet"].get("description", "")[:200]
                video_url = f"https://www.youtube.com/watch?v={video['id']['videoId']}"
                views = int(stat["statistics"].get("viewCount", 0))
                subscribers = int(channel["statistics"].get("subscriberCount", 0))

                if subscribers < 3000:
                    all_results.append({
                        "Title": title,
                        "Description": description,
                        "URL": video_url,
                        "Views": views,
                        "Subscribers": subscribers
                    })

        # Display results
        if all_results:
            st.success(f"✅ Found {len(all_results)} viral videos from smaller channels!")
            for result in all_results:
                st.markdown(
                    f"**🎬 Title:** {result['Title']}  \n"
                    f"📖 **Description:** {result['Description']}  \n"
                    f"🔗 [Watch Video]({result['URL']})  \n"
                    f"👁️ Views: {result['Views']}  \n"
                    f"📢 Subscribers: {result['Subscribers']}"
                )
                st.markdown("---")
        else:
            st.warning("No results from small channels with < 3,000 subs found.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
