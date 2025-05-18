"""
News page for the Finance Board application.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import date, timedelta, datetime
import requests


def render_news_page(ticker, asset_name):
    """
    Render the news page.
    """
    st.header(f"News for {asset_name} ({ticker})")

    # --- Live News Feed ---
    st.subheader("Latest News Headlines")

    # Use the API key directly
    api_key = st.secrets['NEWSAPI_KEY']

    try:
        # Create search phrase - use both ticker and company name with financial keywords for better results
        phrase = f"({ticker} OR \"{asset_name}\") AND (stock OR market OR finance OR earnings OR investor OR financial OR trading OR shares OR investment)"

        # Set date range - last 7 days
        end_date = date.today()
        start_date = end_date - timedelta(days=7)

        # Format dates for API
        from_date = start_date.isoformat()
        to_date = end_date.isoformat()

        # Use direct requests instead of the client library to avoid potential issues
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': phrase,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 50,  # Request more articles to ensure we get enough relevant ones
            'apiKey': api_key
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])

            if articles:
                # Create a dataframe for analysis
                news_df = pd.DataFrame(articles)

                # Display news in a more structured way with improved styling
                st.success(f"Found {len(articles)} news articles related to {asset_name} ({ticker})")
                for article in articles[:15]:  # Display top 15 news items
                    # Safely access article keys with .get() method to avoid KeyError
                    title = article.get('title', 'No title available')
                    published_at = article.get('publishedAt', 'Unknown date')
                    link = article.get('url', '#')
                    description = article.get('description', 'No description found')
                    source = article.get('source', {}).get('name', 'Unknown source')

                    # Format the date nicely
                    try:
                        pub_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                        formatted_date = pub_date.strftime("%Y-%m-%d %H:%M")
                    except:
                        formatted_date = published_at

                    # Create a card-like container for each news item
                    with st.container():
                        st.markdown(f"""
                        <div style="padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                            <h4 style="margin-top: 0;"><a href="{link}" target="_blank" style="color: #3274a1; text-decoration: none; font-weight: 600;">{title}</a></h4>
                        """, unsafe_allow_html=True)

                        # Format the publication time
                        st.markdown(f"""
                        <p style="font-size: 0.85rem; color: #666; margin-bottom: 10px;">
                            <span style="font-weight: 500;">{source}</span> • {formatted_date}
                        </p>
                        """, unsafe_allow_html=True)

                        # Add summary if available
                        if description:
                            st.markdown(f"""
                            <p style="color: #333; margin-bottom: 0;">{description[:300]}{'...' if len(description) > 300 else ''}</p>
                            """, unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

            else:
                # Try a more general search as a fallback
                st.info(f"No specific financial news found for {ticker} ({asset_name}). Trying a broader search...")

                # Create a more general search phrase
                fallback_phrase = f"{ticker} OR \"{asset_name}\""

                # Update the parameters with the fallback phrase
                params['q'] = fallback_phrase

                # Make the fallback request
                fallback_response = requests.get(url, params=params)

                if fallback_response.status_code == 200:
                    fallback_data = fallback_response.json()
                    fallback_articles = fallback_data.get('articles', [])

                    if fallback_articles:
                        # Create a dataframe for analysis
                        news_df = pd.DataFrame(fallback_articles)

                        # Display news in a more structured way with improved styling
                        st.warning(f"Found {len(fallback_articles)} general news articles related to {asset_name} ({ticker})")

                        for article in fallback_articles[:15]:  # Display top 15 news items
                            # Safely access article keys with .get() method to avoid KeyError
                            title = article.get('title', 'No title available')
                            published_at = article.get('publishedAt', 'Unknown date')
                            link = article.get('url', '#')
                            description = article.get('description', 'No description found')
                            source = article.get('source', {}).get('name', 'Unknown source')

                            # Format the date nicely
                            try:
                                pub_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                                formatted_date = pub_date.strftime("%Y-%m-%d %H:%M")
                            except:
                                formatted_date = published_at

                            # Create a card-like container for each news item
                            with st.container():
                                st.markdown(f"""
                                <div style="padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                                    <h4 style="margin-top: 0;"><a href="{link}" target="_blank" style="color: #3274a1; text-decoration: none; font-weight: 600;">{title}</a></h4>
                                """, unsafe_allow_html=True)

                                # Format the publication time
                                st.markdown(f"""
                                <p style="font-size: 0.85rem; color: #666; margin-bottom: 10px;">
                                    <span style="font-weight: 500;">{source}</span> • {formatted_date}
                                </p>
                                """, unsafe_allow_html=True)

                                # Add summary if available
                                if description:
                                    st.markdown(f"""
                                    <p style="color: #333; margin-bottom: 0;">{description[:300]}{'...' if len(description) > 300 else ''}</p>
                                    """, unsafe_allow_html=True)

                                st.markdown("</div>", unsafe_allow_html=True)

                    else:
                        st.info(f"No news articles found for {ticker} ({asset_name}).")
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 8px; background-color: #f8f9fa; text-align: center; margin-top: 20px;">
                            <h3 style="margin-bottom: 15px;">No news found</h3>
                            <p>We couldn't find any news articles for {asset_name} ({ticker}).</p>
                            <p>Try a different ticker symbol or check back later for updates.</p>
                            <p style="font-size: 0.8rem; margin-top: 15px; color: #666;">Search queries tried:</p>
                            <p style="font-size: 0.8rem; color: #666;">1. {phrase}</p>
                            <p style="font-size: 0.8rem; color: #666;">2. {fallback_phrase}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"No financial news articles found for {ticker} ({asset_name}).")
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 8px; background-color: #f8f9fa; text-align: center; margin-top: 20px;">
                        <h3 style="margin-bottom: 15px;">No financial news found</h3>
                        <p>We couldn't find any recent financial news articles for {asset_name} ({ticker}).</p>
                        <p>Try a different ticker symbol or check back later for updates.</p>
                        <p style="font-size: 0.8rem; margin-top: 15px; color: #666;">Search query: {phrase}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error(f"Error fetching news: {response.status_code}")
            if response.status_code == 429:
                st.warning("API rate limit exceeded. Please try again later.")
            elif response.status_code == 401:
                st.warning("API key authentication error. Please check your API key.")
            else:
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 8px; background-color: #f8f9fa; text-align: center; margin-top: 20px;">
                    <h3 style="margin-bottom: 15px;">News data is currently unavailable</h3>
                    <p>We're using NewsAPI to fetch the latest news for this ticker.</p>
                    <p>Error code: {response.status_code}</p>
                    <p>If you continue to see this message, please try another ticker or check back later.</p>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred while fetching news: {str(e)}")
        st.markdown("""
        <div style="padding: 20px; border-radius: 8px; background-color: #f8f9fa; text-align: center; margin-top: 20px;">
            <h3 style="margin-bottom: 15px;">News data is currently unavailable</h3>
            <p>We're using NewsAPI to fetch the latest news for this ticker.</p>
            <p>If you continue to see this message, please try another ticker or check back later.</p>
        </div>
        """, unsafe_allow_html=True)

