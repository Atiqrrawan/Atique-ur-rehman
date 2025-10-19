import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np
import io
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from PIL import Image
import cv2
import pytesseract
from sklearn.decomposition import PCA

# ----------------------- CONFIG -----------------------
# IMPORTANT: Use Streamlit secrets for API keys instead of hardcoding
YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# Initialize YouTube client
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# ----------------------- HELPERS -----------------------

def iso_days_ago(days):
    return (datetime.utcnow() - timedelta(days=days)).isoformat("T") + "Z"

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0

def fetch_search_videos(query, region, published_after, max_results=50):
    req = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        publishedAfter=published_after,
        regionCode=region,
        maxResults=max_results,
        relevanceLanguage='en'
    )
    res = req.execute()
    return res.get('items', [])

def get_video_details(video_ids):
    details = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        resp = youtube.videos().list(
            part='statistics,snippet,contentDetails',
            id=','.join(chunk)
        ).execute()
        for item in resp.get('items', []):
            details[item['id']] = item
    return details

def get_channel_details(channel_ids):
    details = {}
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i:i+50]
        resp = youtube.channels().list(
            part='statistics,snippet',
            id=','.join(chunk)
        ).execute()
        for item in resp.get('items', []):
            details[item['id']] = item
    return details

# ----------------------- Thumbnail Helpers -----------------------

def download_image(url):
    try:
        resp = requests.get(url, timeout=10)
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        return img
    except Exception:
        return None

def dominant_colors(image_pil, n_colors=3):
    img = np.array(image_pil)
    img_small = cv2.resize(img, (150, 150)).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(img_small)
    centers = kmeans.cluster_centers_.astype(int)
    return ['#%02x%02x%02x' % tuple(c) for c in centers]

def detect_faces_and_text(image_pil):
    img = np.array(image_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    text = pytesseract.image_to_string(image_pil)
    return len(faces), text.strip()

# ----------------------- Keyword / AI Helpers -----------------------

def extract_keywords_from_tags(tag_lists):
    all_tags = []
    for tags in tag_lists:
        if tags:
            all_tags.extend([t.lower() for t in tags])
    return all_tags

def cluster_keywords(texts, k=6):
    if not texts:
        return []
    vec = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vec.fit_transform(texts)
    if X.shape[0] < k:
        k = max(1, X.shape[0] // 2)
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    terms = vec.get_feature_names_out()
    clusters = defaultdict(list)
    for i, label in enumerate(km.labels_):
        clusters[label].append(texts[i])
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    summaries = []
    for i in range(km.n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        summaries.append({'cluster': i, 'top_terms': top_terms, 'examples': clusters[i][:6]})
    return summaries

def keyword_trend_predict(dates_counts):
    if len(dates_counts) < 3:
        return None
    items = sorted(dates_counts.items())
    xs = np.array([datetime.fromisoformat(d).toordinal() for d, _ in items]).reshape(-1,1)
    ys = np.array([c for _, c in items])
    model = LinearRegression().fit(xs, ys)
    next_day = np.array([[xs[-1,0] + 1]])
    pred = model.predict(next_day)[0]
    return max(0, pred)

def openai_generate(prompt, max_tokens=150, temperature=0.7):
    if not OPENAI_API_KEY:
        return "[OpenAI key not set — enable in Streamlit secrets]"
    import requests
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role":"user","content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    r = requests.post(url, headers=headers, json=data, timeout=20)
    if r.status_code == 200:
        return r.json()['choices'][0]['message']['content'].strip()
    else:
        return f"[OpenAI error: {r.status_code} — {r.text}]"

# ----------------------- STREAMLIT UI -----------------------

st.set_page_config(page_title='YouTube Viral History AI Assistant', layout='wide')
st.title('🎯 YouTube Viral History AI Assistant')
st.markdown('Analyze viral history videos, thumbnails, keywords, and generate AI ideas/tags.')

# Sidebar filters
st.sidebar.header('Search Filters')
region = st.sidebar.selectbox('Region code', ['US', 'CA'])
min_subs = st.sidebar.number_input('Min channel subscribers', value=40000, step=1000)
days = st.sidebar.slider('Published in last (days)', 1, 30, 10)
min_views = st.sidebar.number_input('Min views', value=100000, step=10000)
query = st.sidebar.text_input('Search query', value='history documentary OR ancient civilization')
max_results = st.sidebar.selectbox('Max results', [25, 50, 75], index=1)

if st.sidebar.button('Run Search'):
    published_after = iso_days_ago(days)
    with st.spinner('Searching YouTube...'):
        items = fetch_search_videos(query, region, published_after, max_results=max_results)
    if not items:
        st.warning("No videos found.")
    else:
        video_ids = [it['id']['videoId'] for it in items if it.get('id') and it['id'].get('videoId')]
        video_details = get_video_details(video_ids)
        channel_ids = list({video_details[vid]['snippet']['channelId'] for vid in video_details})
        channel_details = get_channel_details(channel_ids)

        # Build DataFrame
        rows = []
        tag_lists = []
        tag_date_counts = defaultdict(lambda: defaultdict(int))
        for vid, item in video_details.items():
            snip = item['snippet']
            stats = item.get('statistics', {})
            cid = snip.get('channelId')
            ch = channel_details.get(cid, {})
            subs = safe_int(ch.get('statistics', {}).get('subscriberCount', 0))
            views = safe_int(stats.get('viewCount', 0))
            likes = safe_int(stats.get('likeCount', 0))
            comments = safe_int(stats.get('commentCount', 0))
            published = snip.get('publishedAt', '')[:10]
            tags = item['snippet'].get('tags', [])
            tag_lists.append(tags)
            for t in tags:
                tag_date_counts[t.lower()][published] += 1

            if subs >= min_subs and views >= min_views:
                rows.append({
                    'video_id': vid,
                    'title': snip.get('title'),
                    'channel': snip.get('channelTitle'),
                    'channel_id': cid,
                    'publishedAt': published,
                    'views': views,
                    'likes': likes,
                    'comments': comments,
                    'subs': subs,
                    'tags': tags,
                    'thumbnail': snip.get('thumbnails', {}).get('medium', {}).get('url')
                })
        df = pd.DataFrame(rows)
        st.success(f'Found {len(df)} videos matching filters')
