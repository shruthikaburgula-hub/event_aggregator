import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(__file__))

from data.mock_events import mock_events
from ml.categorizer import load_classifier, categorize_all_events, CATEGORIES, CATEGORY_EMOJI
from ml.semantic_search import load_embedder, embed_events, semantic_search
from ml.crowd_predictor import predict_all

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EventPulse – Your City, Your Vibe",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0D0D0D;
    color: #F0F0F0;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; }

.stApp { background: #0D0D0D; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #141414 !important;
    border-right: 1px solid #222;
}

/* Cards */
.event-card {
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 16px;
    padding: 0;
    margin-bottom: 20px;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
    cursor: pointer;
}
.event-card:hover {
    transform: translateY(-4px);
    border-color: #444;
}
.card-image {
    width: 100%;
    height: 160px;
    object-fit: cover;
}
.card-body { padding: 16px; }
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #F0F0F0;
    margin-bottom: 6px;
    line-height: 1.3;
}
.card-meta {
    font-size: 12px;
    color: #888;
    margin-bottom: 10px;
}
.card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.source-badge {
    background: #222;
    color: #AAA;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 20px;
}
.price-tag {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 14px;
    color: #F0F0F0;
}
.crowd-chip {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #1A0A2E 0%, #0D0D0D 60%);
    border: 1px solid #2A1A4A;
    border-radius: 20px;
    padding: 40px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #7C3AED33, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(90deg, #A78BFA, #F472B6, #FB923C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero-sub { font-size: 15px; color: #888; }

/* Stat chips */
.stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 18px; }
.stat-chip {
    background: #1F1F1F;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 13px;
    color: #CCC;
}
.stat-chip span { font-weight: 700; color: #A78BFA; }

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #F0F0F0;
    margin: 24px 0 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Search bar override */
.stTextInput > div > div > input {
    background: #1A1A1A !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    color: #F0F0F0 !important;
    padding: 12px 16px !important;
    font-size: 15px !important;
}

/* Onboarding form */
.onboard-card {
    background: #141414;
    border: 1px solid #2A2A2A;
    border-radius: 20px;
    padding: 36px;
    max-width: 600px;
    margin: 40px auto;
}
.onboard-title {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #A78BFA, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.onboard-sub { text-align: center; color: #777; font-size: 14px; margin-bottom: 28px; }

/* Multiselect & selectbox */
.stMultiSelect [data-baseweb="tag"] { background: #7C3AED !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #DB2777) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    width: 100% !important;
}

/* Divider */
hr { border-color: #222 !important; }

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
if "onboarded" not in st.session_state:
    st.session_state.onboarded = False
if "prefs" not in st.session_state:
    st.session_state.prefs = {}
if "events" not in st.session_state:
    st.session_state.events = []
if "page" not in st.session_state:
    st.session_state.page = "Discover"


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    classifier = load_classifier()
    embedder = load_embedder()
    return classifier, embedder


# ── Onboarding ────────────────────────────────────────────────────────────────
def show_onboarding():
    st.markdown("""
    <div style='display:flex;justify-content:center;margin-top:30px'>
        <div style='text-align:center'>
            <div style='font-size:52px'>🎯</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='onboard-card'>
        <div class='onboard-title'>Welcome to EventPulse</div>
        <div class='onboard-sub'>Tell us what you love — we'll find your perfect events.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("onboarding_form"):
        st.markdown("#### 🏙️ Your City")
        city = st.selectbox("Select your city", ["Hyderabad", "Mumbai", "Bangalore", "Delhi", "Chennai", "Pune", "Other"])

        st.markdown("#### 🎭 Your Interests *(pick all that apply)*")
        interests = st.multiselect(
            "What kind of events do you enjoy?",
            options=list(CATEGORIES),
            default=["Music & Concerts", "Technology & Hackathons", "Stand-up Comedy"],
        )

        st.markdown("#### 💸 Budget Range")
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min Price (₹)", min_value=0, value=0, step=100)
        with col2:
            max_price = st.number_input("Max Price (₹)", min_value=0, value=2000, step=100)

        st.markdown("#### 📅 Preferred Days")
        days = st.multiselect(
            "When do you usually go out?",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            default=["Friday", "Saturday", "Sunday"],
        )

        submitted = st.form_submit_button("🚀 Let's Find My Events!")
        if submitted:
            if not interests:
                st.error("Please select at least one interest!")
            else:
                st.session_state.prefs = {
                    "city": city,
                    "interests": interests,
                    "min_price": min_price,
                    "max_price": max_price,
                    "days": days,
                }
                st.session_state.onboarded = True
                st.rerun()


# ── Event Card ────────────────────────────────────────────────────────────────
def render_event_card(event):
    crowd = event.get("crowd", {})
    category = event.get("category", "Event")
    emoji = event.get("category_emoji", "🎯")
    color = event.get("category_color", "#888")
    price_str = "Free" if event["price"] == 0 else f"₹{event['price']:,}"
    crowd_label = crowd.get("label", "")
    crowd_color = crowd.get("color", "#888")
    source_url = event.get("source_url", "#")
    btn_label = "Register Free →" if event["price"] == 0 else "Book Tickets →"

    st.markdown(f"""
    <div class='event-card'>
        <img class='card-image' src='{event["image"]}' alt='{event["title"]}'
             onerror="this.src='https://images.unsplash.com/photo-1492684223066-81342ee5ff30?w=400'"/>
        <div class='card-body'>
            <div class='card-title'>{event['title']}</div>
            <div class='card-meta'>📍 {event['venue']} &nbsp;|&nbsp; 📅 {event['date']} &nbsp;|&nbsp; 🕐 {event['time']}</div>
            <div style='font-size:12px;color:#999;line-height:1.5;margin-bottom:10px'>
                {event['description'][:110]}...
            </div>
            <div class='card-footer'>
                <span class='badge' style='background:{color}22;color:{color};border:1px solid {color}44'>
                    {emoji} {category}
                </span>
                <span class='crowd-chip' style='background:{crowd_color}22;color:{crowd_color};border:1px solid {crowd_color}44'>
                    {crowd_label}
                </span>
                <span class='price-tag'>{price_str}</span>
                <span class='source-badge'>{event['source']}</span>
            </div>
            <a href='{source_url}' target='_blank' style='
                display:block;
                margin-top:10px;
                padding:7px 0;
                text-align:center;
                background:linear-gradient(135deg,#7C3AED,#DB2777);
                color:white;
                font-family:Syne,sans-serif;
                font-weight:700;
                font-size:12px;
                border-radius:8px;
                text-decoration:none;
                letter-spacing:0.3px;
            '>{btn_label}</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def show_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:16px 0 8px'>
            <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;
                background:linear-gradient(90deg,#A78BFA,#F472B6);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                🎯 EventPulse
            </div>
            <div style='font-size:12px;color:#555;margin-top:2px'>Your city. Your vibe.</div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        prefs = st.session_state.prefs
        st.markdown(f"**🏙️ City:** {prefs.get('city', 'Not set')}")
        st.markdown(f"**💸 Budget:** ₹{prefs.get('min_price',0)} – ₹{prefs.get('max_price',2000)}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**📌 Navigation**")

        pages = ["🔍 Discover", "📊 Insights", "⚙️ Preferences"]
        for p in pages:
            if st.button(p, key=f"nav_{p}"):
                st.session_state.page = p.split(" ", 1)[1]
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**🧠 AI Models Active**")
        st.markdown("""
        <div style='font-size:12px;color:#777;line-height:2'>
        ✅ DistilBERT Categorizer<br>
        ✅ Sentence Transformer Search<br>
        ✅ Crowd Score Predictor<br>
        ⏳ NCF Recommender <i>(coming soon)</i><br>
        ⏳ CNN Image Tagger <i>(coming soon)</i>
        </div>
        """, unsafe_allow_html=True)


# ── Discover Page ─────────────────────────────────────────────────────────────
def show_discover(events):
    prefs = st.session_state.prefs
    city = prefs.get("city", "Your City")

    st.markdown(f"""
    <div class='hero'>
        <div class='hero-title'>What's happening in {city}? 🔥</div>
        <div class='hero-sub'>AI-curated events from Eventbrite, Meetup, Ticketmaster, Google & more</div>
        <div class='stat-row'>
            <div class='stat-chip'><span>{len(events)}</span> Events Found</div>
            <div class='stat-chip'><span>6</span> Sources</div>
            <div class='stat-chip'><span>3</span> AI Models</div>
            <div class='stat-chip'>Updated <span>Live</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Search bar
    query = st.text_input("", placeholder="🔍  Search anything... 'chill music event' or 'free tech meetup this weekend'")

    # Filters row
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        cat_filter = st.multiselect("Category", options=list(CATEGORIES), default=[], placeholder="All categories")
    with col2:
        crowd_filter = st.multiselect("Crowd Level", ["🟢 Chill", "🟡 Moderate", "🔴 Packed"], default=[], placeholder="Any crowd")
    with col3:
        price_filter = st.selectbox("Price", ["Any", "Free only", "Under ₹500", "Under ₹1000", "Under ₹2000"])

    st.markdown("<hr>", unsafe_allow_html=True)

    # Apply semantic search or filters
    with st.spinner("🧠 AI is thinking..."):
        if query.strip():
            embedder = st.session_state.get("embedder")
            embeddings = st.session_state.get("event_embeddings")
            if embedder and embeddings is not None:
                filtered = semantic_search(query, events, embedder, embeddings)
            else:
                filtered = events
        else:
            filtered = events

    # Apply category filter
    if cat_filter:
        filtered = [e for e in filtered if e.get("category") in cat_filter]

    # Apply price filter
    if price_filter == "Free only":
        filtered = [e for e in filtered if e["price"] == 0]
    elif price_filter == "Under ₹500":
        filtered = [e for e in filtered if e["price"] < 500]
    elif price_filter == "Under ₹1000":
        filtered = [e for e in filtered if e["price"] < 1000]
    elif price_filter == "Under ₹2000":
        filtered = [e for e in filtered if e["price"] < 2000]

    # Apply crowd filter
    if crowd_filter:
        crowd_map = {"🟢 Chill": "Low", "🟡 Moderate": "Medium", "🔴 Packed": "High"}
        allowed = [crowd_map[c] for c in crowd_filter]
        filtered = [e for e in filtered if e.get("crowd", {}).get("level") in allowed]

    # City filter from preferences
    city_filtered = [e for e in filtered if e.get("city", "").lower() == city.lower()]
    other_cities = [e for e in filtered if e.get("city", "").lower() != city.lower()]

    # Results
    if query.strip():
        st.markdown(f"<div class='section-header'>🔍 Search Results for \"{query}\"</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='section-header'>🏙️ Events in {city}</div>", unsafe_allow_html=True)

    if not city_filtered and not filtered:
        st.info("No events found. Try a different search or adjust your filters!")
    else:
        display_events = city_filtered if city_filtered else filtered
        cols = st.columns(3)
        for i, event in enumerate(display_events):
            with cols[i % 3]:
                render_event_card(event)

    if other_cities and not query.strip():
        st.markdown("<div class='section-header'>🌍 Events in Other Cities</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, event in enumerate(other_cities):
            with cols[i % 3]:
                render_event_card(event)


# ── Insights Page ─────────────────────────────────────────────────────────────
def show_insights(events):
    st.markdown("<div class='hero-title' style='font-size:28px'>📊 City Insights</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#777;margin-bottom:24px'>AI-generated insights from all events this month</div>", unsafe_allow_html=True)

    from collections import Counter

    # Category breakdown
    cats = [e.get("category", "Unknown") for e in events]
    cat_counts = Counter(cats)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🏷️ Top Event Categories**")
        for cat, count in cat_counts.most_common(5):
            emoji = CATEGORY_EMOJI.get(cat, "🎯")
            pct = int(count / len(events) * 100)
            st.markdown(f"""
            <div style='margin-bottom:10px'>
                <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px'>
                    <span>{emoji} {cat}</span><span style='color:#888'>{count} events</span>
                </div>
                <div style='background:#222;border-radius:6px;height:6px'>
                    <div style='background:linear-gradient(90deg,#7C3AED,#DB2777);width:{pct}%;height:6px;border-radius:6px'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**👥 Crowd Distribution**")
        crowd_levels = [e.get("crowd", {}).get("level", "Unknown") for e in events]
        crowd_counts = Counter(crowd_levels)
        colors = {"Low": "#6BCB77", "Medium": "#FFD93D", "High": "#FF6B6B"}
        labels = {"Low": "🟢 Chill", "Medium": "🟡 Moderate", "High": "🔴 Packed"}
        for level in ["Low", "Medium", "High"]:
            count = crowd_counts.get(level, 0)
            pct = int(count / len(events) * 100) if events else 0
            st.markdown(f"""
            <div style='margin-bottom:10px'>
                <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px'>
                    <span>{labels[level]}</span><span style='color:#888'>{count} events</span>
                </div>
                <div style='background:#222;border-radius:6px;height:6px'>
                    <div style='background:{colors[level]};width:{pct}%;height:6px;border-radius:6px'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Source breakdown
    st.markdown("**📡 Events by Source**")
    sources = [e.get("source", "Unknown") for e in events]
    source_counts = Counter(sources)
    src_cols = st.columns(len(source_counts))
    for i, (src, count) in enumerate(source_counts.items()):
        with src_cols[i]:
            st.markdown(f"""
            <div style='background:#1A1A1A;border:1px solid #2A2A2A;border-radius:12px;padding:16px;text-align:center'>
                <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#A78BFA'>{count}</div>
                <div style='font-size:12px;color:#777;margin-top:4px'>{src}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Best time tip
    st.markdown("**💡 Smart Tips**")
    free_events = [e for e in events if e["price"] == 0]
    cheap_events = [e for e in events if 0 < e["price"] <= 500]
    chill_events = [e for e in events if e.get("crowd", {}).get("level") == "Low"]

    tip_cols = st.columns(3)
    with tip_cols[0]:
        st.markdown(f"""
        <div style='background:#1A0A2E;border:1px solid #3A1A5E;border-radius:12px;padding:16px'>
            <div style='font-size:24px'>🆓</div>
            <div style='font-family:Syne,sans-serif;font-weight:700;margin:6px 0'>{len(free_events)} Free Events</div>
            <div style='font-size:12px;color:#888'>Great events at zero cost this month</div>
        </div>
        """, unsafe_allow_html=True)
    with tip_cols[1]:
        st.markdown(f"""
        <div style='background:#0A1A0A;border:1px solid #1A3A1A;border-radius:12px;padding:16px'>
            <div style='font-size:24px'>🟢</div>
            <div style='font-family:Syne,sans-serif;font-weight:700;margin:6px 0'>{len(chill_events)} Low-Crowd Events</div>
            <div style='font-size:12px;color:#888'>Perfect for a relaxed outing</div>
        </div>
        """, unsafe_allow_html=True)
    with tip_cols[2]:
        st.markdown(f"""
        <div style='background:#1A0A0A;border:1px solid #3A1A1A;border-radius:12px;padding:16px'>
            <div style='font-size:24px'>💰</div>
            <div style='font-family:Syne,sans-serif;font-weight:700;margin:6px 0'>{len(cheap_events)} Budget-Friendly</div>
            <div style='font-size:12px;color:#888'>Under ₹500, worth every rupee</div>
        </div>
        """, unsafe_allow_html=True)


# ── Preferences Page ──────────────────────────────────────────────────────────
def show_preferences():
    st.markdown("<div style='font-family:Syne,sans-serif;font-size:24px;font-weight:800;margin-bottom:20px'>⚙️ Update Preferences</div>", unsafe_allow_html=True)
    prefs = st.session_state.prefs

    with st.form("prefs_form"):
        city = st.selectbox("🏙️ City", ["Hyderabad", "Mumbai", "Bangalore", "Delhi", "Chennai", "Pune", "Other"],
                            index=["Hyderabad", "Mumbai", "Bangalore", "Delhi", "Chennai", "Pune", "Other"].index(prefs.get("city", "Hyderabad")))
        interests = st.multiselect("🎭 Interests", options=list(CATEGORIES), default=prefs.get("interests", []))
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min Price (₹)", value=prefs.get("min_price", 0), step=100)
        with col2:
            max_price = st.number_input("Max Price (₹)", value=prefs.get("max_price", 2000), step=100)

        if st.form_submit_button("💾 Save Preferences"):
            st.session_state.prefs = {**prefs, "city": city, "interests": interests,
                                       "min_price": min_price, "max_price": max_price}
            st.success("✅ Preferences saved!")
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.onboarded:
        show_onboarding()
        return

    # Load models + process events once
    with st.spinner("🤖 Loading AI models..."):
        classifier, embedder = load_models()

    if not st.session_state.events:
        with st.spinner("🏷️ Categorizing events with DistilBERT..."):
            events = categorize_all_events(mock_events, classifier)
        with st.spinner("👥 Predicting crowd levels..."):
            events = predict_all(events)
        st.session_state.events = events

        with st.spinner("🔗 Building semantic search index..."):
            embeddings = embed_events(events, embedder)
            st.session_state.event_embeddings = embeddings
            st.session_state.embedder = embedder

    events = st.session_state.events
    show_sidebar()

    page = st.session_state.page
    if page == "Discover" or page not in ["Insights", "Preferences"]:
        show_discover(events)
    elif page == "Insights":
        show_insights(events)
    elif page == "Preferences":
        show_preferences()


if __name__ == "__main__":
    main()
