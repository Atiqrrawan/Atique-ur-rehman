import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import io
import requests
from PIL import Image
import cv2
import pytesseract

# Machine Learning / Text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from collections import defaultdict

# YouTube API
from googleapiclient.discovery import build

# ----------------------- CONFIG -----------------------
# Streamlit secrets
YOUTUBE_API_KEY = st.secrets["AIzaSyA31_v0xRKf5_KSM_I7wCbUNy8KKKZdRnU"]
OPENAI_API_KEY = st.secrets.get("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", "")

# Initialize YouTube client safely
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# ----------------------- HELPERS -----------------------

def iso_days_ago(days):
    return (datetime.utcnow() - timedelta(days=days)).isoformat("T") + "Z"

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0

# ------------------- YouTube API Helpers -------------------

def fetch_search_videos(query, region, published_after, max_results=50):
    """Safe fetch YouTube videos with TransportError handling"""
    try:
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
    except Exception as e:
        st.error(f"YouTube API error: {e}")
        return []

def get_video_details(video_ids):
    details = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        try:
            resp = youtube.videos().list(part='statistics,snippet,contentDetails', id=','.join(chunk)).execute()
            for item in resp.get('items', []):
                vid = item['id']
                details[vid] = item
        except Exception as e:
            st.warning(f"Video details fetch error: {e}")
    return details

def get_channel_details(channel_ids):
    details = {}
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i:i+50]
        try:
            resp = youtube.channels().list(part='statistics,snippet', id=','.join(chunk)).execute()
            for item in resp.get('items', []):
                cid = item['id']
                details[cid] = item
        except Exception as e:
            st.warning(f"Channel details fetch error: {e}")
    return details

# ------------------- Thumbnail / Image Helpers -------------------

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
    hexes = ['#%02x%02x%02x' % tuple(c) for c in centers]
    return hexes

def detect_faces_and_text(image_pil):
    img = np.array(image_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    text = pytesseract.image_to_string(image_pil)
    return len(faces), text.strip()

# ------------------- Keyword Helpers -------------------

def extract_keywords_from_tags(tag_lists):
    all_tags = []
    for tags in tag_lists:
        if not tags:
            continue
        all_tags.extend(tags)
    return [t.lower() for t in all_tags]

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
    cluster_summaries = []
    for i in range(km.n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        cluster_summaries.append({'cluster': i, 'top_terms': top_terms, 'examples': clusters[i][:6]})
    return cluster_summaries

def keyword_trend_predict(dates_counts):
    if len(dates_counts) < 3:
        return None
    items = sorted(dates_counts.items())
    xs = np.array([(datetime.fromisoformat(d)).toordinal() for d, _ in items]).reshape(-1,1)
    ys = np.array([c for _, c in items])
    model = LinearRegression().fit(xs, ys)
    next_day = np.array([[xs[-1,0] + 1]])
    pred = model.predict(next_day)[0]
    return max(0, pred)

# ------------------- OpenAI AI Generator -------------------

def openai_generate(prompt, max_tokens=150, temperature=0.7):
    if not OPENAI_API_KEY:
        return "[OpenAI key not set — set OPENAI_API_KEY in Streamlit secrets]"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role":"user","content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        r = requests.post(url, headers=headers, json=data, timeout=20)
        if r.status_code == 200:
            j = r.json()
            return j['choices'][0]['message']['content'].strip()
        else:
            return f"[OpenAI error: {r.status_code} — {r.text}]"
    except Exception as e:
        return f"[OpenAI request error: {e}]"

# ------------------- STREAMLIT UI -------------------

st.set_page_config(page_title='YouTube Viral History AI Assistant', layout='wide')
st.title('🎯 YouTube Viral History AI Assistant')
st.markdown('Find viral English history videos, analyze thumbnails, cluster tags, and generate AI ideas.')

# Sidebar filters
st.sidebar.header('Search Filters')
region = st.sidebar.selectbox('Region code', ['US', 'CA'])
min_subs = st.sidebar.number_input('Min channel subscribers', value=40000, step=1000)
days = st.sidebar.slider('Published in last (days)', 1, 30, 10)
min_views = st.sidebar.number_input('Min views', value=100000, step=10000)
query = st.sidebar.text_input('Search query', value='history documentary OR ancient civilization OR historical events')
max_results = st.sidebar.selectbox('Max results (search API)', [25,50,75], index=1)

if st.sidebar.button('Run Search'):
    published_after = iso_days_ago(days)
    with st.spinner('Searching YouTube...'):
        items = fetch_search_videos(query, region, published_after, max_results=max_results)

    if not items:
        st.warning('No search results returned.')
    else:
        video_ids = [it['id']['videoId'] for it in items if it.get('id') and it['id'].get('videoId')]
        video_details = get_video_details(video_ids)
        channel_ids = list({video_details[vid]['snippet']['channelId'] for vid in video_details})
        channel_details = get_channel_details(channel_ids)

        rows = []
        tag_lists = []
        tag_date_counts = defaultdict(lambda: defaultdict(int))

        for vid in video_details:
            item = video_details[vid]
            snip = item['snippet']
            stats = item.get('statistics', {})
            cid = snip.get('channelId')
            ch = channel_details.get(cid, {})
            subs = safe_int(ch.get('statistics', {}).get('subscriberCount',0))
            views = safe_int(stats.get('viewCount',0))
            likes = safe_int(stats.get('likeCount',0))
            comments = safe_int(stats.get('commentCount',0))
            published = snip.get('publishedAt','')[:10]
            tags = snip.get('tags',[])
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

        # Show top channels
        st.subheader('🏆 Top Channels')
        if not df.empty:
            agg = df.groupby('channel').agg({'views':'sum','subs':'max','video_id':'count'}).reset_index().sort_values('views', ascending=False)
            agg = agg.rename(columns={'video_id':'videos_found'})
            st.dataframe(agg.head(10))

        # Show video details with thumbnails
        for idx, r in df.iterrows():
            with st.expander(f"{r['title']} — {r['channel']} ({r['views']:,} views)"):
                cols = st.columns([1,2])
                with cols[0]:
                    if r['thumbnail']:
                        img = download_image(r['thumbnail'])
                        if img:
                            st.image(img, width=320)
                            faces, text = detect_faces_and_text(img)
                            colors = dominant_colors(img, n_colors=3)
                            st.write('Faces detected:', faces)
                            st.write('Dominant colors:', colors)
                            st.write('Thumbnail text sample:', (text[:200]+'...') if text else 'No text')
                with cols[1]:
                    st.markdown(f"**Title:** {r['title']}")
                    st.markdown(f"**Channel:** {r['channel']} — **Subs:** {r['subs']:,}")
                    st.markdown(f"**Published:** {r['publishedAt']} | **Views:** {r['views']:,} | **Likes:** {r['likes']:,} | **Comments:** {r['comments']:,}")
                    st.markdown('**Tags:** ' + (', '.join(r['tags']) if r['tags'] else 'No tags'))
                    st.markdown(f"[Watch on YouTube](https://www.youtube.com/watch?v={r['video_id']})")

        # Keyword analysis
        st.subheader('🔎 Keyword & Tag Analysis')
        flat_tags = extract_keywords_from_tags(tag_lists)
        if flat_tags:
            counter = Counter(flat_tags)
            top_tags = counter.most_common(40)
            st.dataframe(pd.DataFrame(top_tags, columns=['tag','count']))
            clusters = cluster_keywords([' '.join(t.split()) for t in flat_tags], k=6)
            st.write('Keyword Clusters:')
            for c in clusters:
                st.markdown(f"**Cluster {c['cluster']}** — Top terms: {', '.join(c['top_terms'][:8])}")

        # Short-term trend prediction
        st.subheader('📈 Short-term Trend Prediction')
        preds = []
        for tag, cnt in Counter(flat_tags).most_common(10):
            date_counts = tag_date_counts[tag]
            if date_counts:
                pred = keyword_trend_predict(date_counts)
                preds.append((tag, cnt, pred))
        if preds:
            st.dataframe(pd.DataFrame(preds, columns=['tag','recent_count','predicted_next_day_count']))

        # AI Video Idea Generator
        st.subheader('🤖 AI Video Idea Generator')
        seed_keywords = st.text_input('Seed keywords (comma separated)', value=','.join([t for t,_ in Counter(flat_tags).most_common(8)]))
        if st.button('Generate AI Video Idea'):
            prompt = f"You are an expert YouTube content strategist.\nSeed keywords: {seed_keywords}.\nProvide: (1) catchy video title, (2) 2-line hook, (3) 120-200 word SEO description, (4) 15 suggested tags, (5) 3 thumbnail ideas. Format as JSON."
            ai_out = openai_generate(prompt, max_tokens=400)
            st.code(ai_out)

        # CSV download
        st.subheader('⬇ Export Data')
        if not df.empty:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button('Download CSV', csv, 'youtube_history_videos.csv', 'text/csv')
