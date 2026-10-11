
import re
import html
import requests
import streamlit as st
import joblib
import xml.etree.ElementTree as ET

from urllib.parse import quote_plus
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="News Credibility Explorer",
    page_icon="📰",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL AND VECTORIZER
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("headline_news_model.pkl")
    vectorizer = joblib.load("headline_tfidf_vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error(
        "Headline model files not found. Keep "
        "headline_news_model.pkl and "
        "headline_tfidf_vectorizer.pkl in the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the headline model: {error}")
    st.stop()


# --------------------------------------------------
# TEXT CLEANING
# Must match the training script
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# RELEVANCE SCORING
# Compares important words in the claim and result title
# --------------------------------------------------
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of",
    "in", "on", "at", "for", "from", "by", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its",
    "as", "after", "before", "about", "into", "over",
    "under", "has", "have", "had", "will", "would",
    "could", "should", "first"
}


def important_words(text):
    words = re.findall(r"[a-z]+", text.lower())
    return {
        word for word in words
        if len(word) > 2 and word not in STOP_WORDS
    }


def relevance_score(query, title):
    query_words = important_words(query)
    title_words = important_words(title)

    if not query_words or not title_words:
        return 0.0

    overlap = len(query_words.intersection(title_words))
    return overlap / len(query_words)


# --------------------------------------------------
# PUBLICATION DATE PARSER
# --------------------------------------------------
def publication_timestamp(date_text):
    try:
        date_value = parsedate_to_datetime(date_text)

        if date_value.tzinfo is None:
            date_value = date_value.replace(tzinfo=timezone.utc)

        return date_value.timestamp()

    except (TypeError, ValueError, OverflowError):
        return 0.0


def readable_date(date_text):
    try:
        date_value = parsedate_to_datetime(date_text)
        return date_value.strftime("%d %b %Y, %H:%M UTC")
    except (TypeError, ValueError, OverflowError):
        return "Date unavailable"


# --------------------------------------------------
# GOOGLE NEWS RSS SEARCH
# --------------------------------------------------
def search_related_news(query, limit=10, domain=None):
    search_query = query[:220]

    if domain:
        search_query += f" site:{domain}"

    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(search_query)
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    articles = []

    for item in root.findall(".//item"):
        title = html.unescape(
            item.findtext("title", "Untitled article")
        )
        link = item.findtext("link", "")
        published = item.findtext(
            "pubDate",
            ""
        )

        source_element = item.find("source")
        source = (
            html.unescape(source_element.text)
            if source_element is not None and source_element.text
            else "Publisher not specified"
        )

        if not link:
            continue

        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "source": source,
            "timestamp": publication_timestamp(published),
            "relevance": relevance_score(query, title)
        })

        if len(articles) >= limit:
            break

    # Prioritize relevance first, then newer publication dates.
    articles.sort(
        key=lambda article: (
            article["relevance"],
            article["timestamp"]
        ),
        reverse=True
    )

    return articles


# --------------------------------------------------
# OFFICIAL-DOMAIN SEARCH
# --------------------------------------------------
def search_official_sources(query):
    domains = [
        ("Chennai Metro Rail", "chennaimetrorail.org"),
        ("Press Information Bureau (PIB)", "pib.gov.in")
    ]

    results = []
    seen_links = set()

    for source_name, domain in domains:
        try:
            articles = search_related_news(
                query,
                limit=10,
                domain=domain
            )

            for article in articles:
                link = article["link"]

                if link in seen_links:
                    continue

                seen_links.add(link)
                article["official_source_group"] = source_name
                results.append(article)

        except requests.RequestException:
            st.warning(
                f"Search for {source_name} is unavailable right now."
            )
        except ET.ParseError:
            st.warning(
                f"Could not read search results for {source_name}."
            )

    results.sort(
        key=lambda article: (
            article["relevance"],
            article["timestamp"]
        ),
        reverse=True
    )

    return results


# --------------------------------------------------
# DISPLAY ARTICLES
# --------------------------------------------------
def display_articles(articles):
    if not articles:
        st.info(
            "No matching results were found. "
            "A missing result does not prove the claim is false."
        )
        return

    for number, article in enumerate(articles, start=1):
        st.markdown(
            f"**{number}. [{article['title']}]"
            f"({article['link']})**"
        )

        st.caption(
            f"Publisher: {article['source']} | "
            f"Published: {readable_date(article['published'])} | "
            f"Title match: {article['relevance']:.0%}"
        )


# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("📰 News Credibility Explorer")

st.write(
    "Explore a news claim using a machine-learning model "
    "and related reporting."
)

st.warning(
    "Educational tool only. The model can make mistakes. "
    "Search results and model scores do not automatically "
    "establish whether a claim is true or false."
)


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
st.subheader("1. Enter the news")

headline = st.text_input(
    "News headline",
    placeholder="Enter the headline here"
)

article_body = st.text_area(
    "Article body (optional)",
    placeholder="Paste the article text here if available...",
    height=180
)

check_button = st.button(
    "Analyze News",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------
if check_button:
    combined_text = headline.strip()

    if not combined_text:
        st.warning("Please enter a headline or article body.")

    else:
        cleaned_text = clean_text(combined_text)

        if not cleaned_text:
            st.warning(
                "Please enter text containing English letters."
            )

        else:
            # ------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------
            news_vector = vectorizer.transform([cleaned_text])
            prediction = model.predict(news_vector)[0]
            probabilities = model.predict_proba(news_vector)[0]

            class_scores = {
                int(label): float(score)
                for label, score in zip(
                    model.classes_,
                    probabilities
                )
            }

            fake_score = class_scores.get(0, 0.0) * 100
            real_score = class_scores.get(1, 0.0) * 100

            st.divider()
            st.subheader("2. Model prediction")

            if int(prediction) == 0:
                st.error("Model prediction: FAKE NEWS")
            else:
                st.success("Model prediction: REAL NEWS")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Fake-class score", f"{fake_score:.2f}%")
                st.progress(
                    min(max(fake_score / 100, 0.0), 1.0)
                )

            with col2:
                st.metric("Real-class score", f"{real_score:.2f}%")
                st.progress(
                    min(max(real_score / 100, 0.0), 1.0)
                )

            st.caption(
                "These scores reflect the model's output, not the "
                "verified probability that the news is false or true."
            )

            # ------------------------------------------
            # EVIDENCE STATUS
            # ------------------------------------------
            st.divider()
            st.subheader("3. Evidence status")

            st.info(
                "NOT YET VERIFIED. The model prediction is separate "
                "from the evidence search. Open the articles and "
                "check whether they directly support the specific "
                "claim, including its date and details."
            )

            search_query = headline.strip() or combined_text[:200]

            # ------------------------------------------
            # RELATED NEWS
            # ------------------------------------------
            st.subheader("4. Related news articles")

            with st.spinner("Searching and ranking related articles..."):
                try:
                    related_articles = search_related_news(
                        search_query,
                        limit=10
                    )

                    display_articles(related_articles)

                except requests.RequestException:
                    st.warning(
                        "Related-news search is temporarily unavailable. "
                        "Please try again later."
                    )
                except ET.ParseError:
                    st.warning(
                        "The related-news service returned unreadable data."
                    )

            # ------------------------------------------
            # OFFICIAL SOURCES
            # ------------------------------------------
            st.subheader("5. Selected official sources")

            st.write(
                "Results are searched from selected official domains "
                "and ranked by title match and publication date. "
                "A high title match does not confirm that a claim is true."
            )

            with st.spinner("Searching official domains..."):
                official_articles = search_official_sources(
                    search_query
                )

            if official_articles:
                for source_name in [
                    "Press Information Bureau (PIB)",
                    "Chennai Metro Rail"
                ]:
                    matching_articles = [
                        article
                        for article in official_articles
                        if article.get("official_source_group")
                        == source_name
                    ]

                    with st.expander(
                        source_name,
                        expanded=True
                    ):
                        display_articles(matching_articles)

            else:
                st.info(
                    "No results were found from the selected domains. "
                    "This does not prove the claim is false."
                )

            # ------------------------------------------
            # FINAL REMINDER
            # ------------------------------------------
            st.divider()
            st.subheader("6. Before drawing a conclusion")

            st.markdown(
                """
                - Read the full article, not just its headline.
                - Check the date and the context.
                - Look for independent reporting from reputable outlets.
                - Check whether an official source confirms the exact claim.
                - Remember that a missing result is not proof of a false claim.
                """
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.caption(
    "News Credibility Explorer | Experimental machine-learning "
    "predictions should not be the sole basis for judging news."
)
