import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="AutoSense - Automotive Intelligence Platform",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f0f8ff;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .brand-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample data for demonstration
@st.cache_data
def generate_sample_data():
    # Brand data
    brands = ['Tesla', 'BMW', 'Mercedes', 'Audi', 'Toyota', 'Honda', 'Ford', 'Chevrolet', 'Volkswagen', 'Nissan']
    
    # Generate sentiment data
    sentiment_data = []
    for brand in brands:
        for i in range(30):  # 30 days of data
            date = datetime.now() - timedelta(days=i)
            sentiment_data.append({
                'brand': brand,
                'date': date,
                'positive': random.randint(40, 80),
                'negative': random.randint(5, 25),
                'neutral': random.randint(15, 35),
                'mention_count': random.randint(100, 1000),
                'engagement_rate': random.uniform(2.5, 8.5)
            })
    
    # Generate topic data
    topics = ['Performance', 'Design', 'Price', 'Safety', 'Technology', 'Comfort', 'Fuel Economy', 'Reliability']
    topic_data = []
    for brand in brands[:5]:  # Top 5 brands
        for topic in topics:
            topic_data.append({
                'brand': brand,
                'topic': topic,
                'mention_count': random.randint(50, 500),
                'sentiment_score': random.uniform(-1, 1),
                'trend': random.choice(['up', 'down', 'stable'])
            })
    
    # Generate competitor analysis data
    competitor_data = []
    for brand in brands:
        competitor_data.append({
            'brand': brand,
            'market_share': random.uniform(5, 25),
            'sentiment_score': random.uniform(0.3, 0.8),
            'mention_volume': random.randint(1000, 10000),
            'engagement_rate': random.uniform(3, 9),
            'top_keywords': random.sample(['innovative', 'reliable', 'expensive', 'stylish', 'efficient', 'powerful'], 3)
        })
    
    return pd.DataFrame(sentiment_data), pd.DataFrame(topic_data), pd.DataFrame(competitor_data)

# Initialize data
sentiment_df, topic_df, competitor_df = generate_sample_data()

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/white?text=AutoSense", width=200)
    st.markdown("### 🚗 Navigation")
    
    selected_page = st.selectbox(
        "Select Dashboard",
        ["Overview", "Sentiment Analysis", "Competitive Intelligence", "Topic Analysis", "Real-time Monitoring", "Reports & Insights"]
    )
    
    st.markdown("### 🎯 Filters")
    selected_brands = st.multiselect(
        "Select Brands",
        sentiment_df['brand'].unique(),
        default=['Tesla', 'BMW', 'Mercedes']
    )
    
    date_range = st.date_input(
        "Date Range",
        value=[datetime.now() - timedelta(days=30), datetime.now()],
        max_value=datetime.now()
    )
    
    st.markdown("### 📊 Quick Stats")
    total_mentions = sentiment_df[sentiment_df['brand'].isin(selected_brands)]['mention_count'].sum()
    avg_sentiment = sentiment_df[sentiment_df['brand'].isin(selected_brands)]['positive'].mean()
    
    st.metric("Total Mentions", f"{total_mentions:,}")
    st.metric("Avg Positive Sentiment", f"{avg_sentiment:.1f}%")

# Main content area
st.markdown('<h1 class="main-header">AutoSense Intelligence Platform</h1>', unsafe_allow_html=True)

if selected_page == "Overview":
    st.markdown("## 📈 Executive Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Brands Monitored",
            "10",
            "2 new this month"
        )
    
    with col2:
        st.metric(
            "Daily Mentions",
            "25.4K",
            "12% increase"
        )
    
    with col3:
        st.metric(
            "Sentiment Score",
            "72.3%",
            "5.2% improvement"
        )
    
    with col4:
        st.metric(
            "Market Coverage",
            "95%",
            "Social platforms"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sentiment Trend (Last 30 Days)")
        filtered_data = sentiment_df[sentiment_df['brand'].isin(selected_brands)]
        daily_sentiment = filtered_data.groupby('date').agg({
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean'
        }).reset_index()
        
        fig = px.line(daily_sentiment, x='date', y=['positive', 'negative', 'neutral'],
                     title="Sentiment Distribution Over Time",
                     color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Brand Performance Ranking")
        brand_performance = sentiment_df.groupby('brand').agg({
            'positive': 'mean',
            'mention_count': 'sum'
        }).reset_index()
        brand_performance['score'] = brand_performance['positive'] * 0.7 + (brand_performance['mention_count'] / 1000) * 0.3
        brand_performance = brand_performance.sort_values('score', ascending=False)
        
        fig = px.bar(brand_performance.head(8), x='brand', y='score',
                    title="Overall Performance Score",
                    color='score', color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights section
    st.markdown("## 💡 AI-Powered Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <h4>📈 Trending Up</h4>
            <p><strong>Tesla</strong> mentions increased 23% this week, driven by Model Y discussions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
            <h4>⚠️ Attention Needed</h4>
            <p><strong>BMW</strong> negative sentiment spike around pricing concerns for new X7</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="insight-box">
            <h4>🎯 Opportunity</h4>
            <p><strong>Electric vehicles</strong> discussion volume up 45% - prime time for EV marketing</p>
        </div>
        """, unsafe_allow_html=True)

elif selected_page == "Sentiment Analysis":
    st.markdown("## 😊 Sentiment Analysis Dashboard")
    
    # Sentiment overview
    col1, col2, col3 = st.columns(3)
    
    filtered_data = sentiment_df[sentiment_df['brand'].isin(selected_brands)]
    avg_positive = filtered_data['positive'].mean()
    avg_negative = filtered_data['negative'].mean()
    avg_neutral = filtered_data['neutral'].mean()
    
    with col1:
        st.metric("Positive Sentiment", f"{avg_positive:.1f}%", "2.3%")
    with col2:
        st.metric("Negative Sentiment", f"{avg_negative:.1f}%", "-1.2%")
    with col3:
        st.metric("Neutral Sentiment", f"{avg_neutral:.1f}%", "-1.1%")
    
    # Detailed sentiment analysis
    tab1, tab2, tab3 = st.tabs(["📊 Sentiment Trends", "🎭 Emotion Analysis", "📝 Sample Mentions"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment by brand
            brand_sentiment = filtered_data.groupby('brand').agg({
                'positive': 'mean',
                'negative': 'mean',
                'neutral': 'mean'
            }).reset_index()
            
            fig = px.bar(brand_sentiment, x='brand', y=['positive', 'negative', 'neutral'],
                        title="Sentiment Distribution by Brand",
                        barmode='stack',
                        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment correlation with mentions
            fig = px.scatter(filtered_data, x='mention_count', y='positive',
                           color='brand', size='engagement_rate',
                           title="Sentiment vs Mention Volume",
                           hover_data=['brand'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🎭 Emotional Sentiment Breakdown")
        
        emotions = ['Joy', 'Trust', 'Surprise', 'Anticipation', 'Anger', 'Fear', 'Sadness', 'Disgust']
        emotion_data = []
        
        for brand in selected_brands:
            for emotion in emotions:
                emotion_data.append({
                    'brand': brand,
                    'emotion': emotion,
                    'intensity': random.uniform(0.1, 0.9)
                })
        
        emotion_df = pd.DataFrame(emotion_data)
        
        fig = px.bar(emotion_df, x='emotion', y='intensity', color='brand',
                    title="Emotional Intensity by Brand",
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📝 Recent High-Impact Mentions")
        
        sample_mentions = [
            {"brand": "Tesla", "text": "The new Model S is absolutely incredible! The acceleration is mind-blowing.", "sentiment": "Positive", "engagement": 1247},
            {"brand": "BMW", "text": "Disappointed with the build quality of my new BMW. Expected better for the price.", "sentiment": "Negative", "engagement": 892},
            {"brand": "Mercedes", "text": "Mercedes continues to set the standard for luxury vehicles. Impressive!", "sentiment": "Positive", "engagement": 634},
            {"brand": "Tesla", "text": "Tesla's customer service needs major improvement. Waiting weeks for support.", "sentiment": "Negative", "engagement": 445},
            {"brand": "Audi", "text": "The interior design of the new A8 is simply stunning. Well done Audi!", "sentiment": "Positive", "engagement": 523}
        ]
        
        for mention in sample_mentions:
            if mention['brand'] in selected_brands:
                sentiment_color = "#2ecc71" if mention['sentiment'] == "Positive" else "#e74c3c"
                st.markdown(f"""
                <div style="border-left: 4px solid {sentiment_color}; padding: 1rem; margin: 1rem 0; background: #f8f9fa;">
                    <strong>{mention['brand']}</strong> | {mention['sentiment']} | {mention['engagement']} engagements<br>
                    <em>"{mention['text']}"</em>
                </div>
                """, unsafe_allow_html=True)

elif selected_page == "Competitive Intelligence":
    st.markdown("## 🏁 Competitive Intelligence")
    
    # Market positioning
    st.subheader("📍 Market Positioning Matrix")
    
    fig = px.scatter(competitor_df, x='market_share', y='sentiment_score',
                    size='mention_volume', color='brand',
                    title="Brand Positioning: Market Share vs Sentiment",
                    hover_data=['engagement_rate'])
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Competitive analysis table
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Competitive Metrics")
        display_df = competitor_df.copy()
        display_df['sentiment_score'] = display_df['sentiment_score'].round(3)
        display_df['market_share'] = display_df['market_share'].round(1)
        display_df['engagement_rate'] = display_df['engagement_rate'].round(2)
        
        st.dataframe(display_df[['brand', 'market_share', 'sentiment_score', 'mention_volume', 'engagement_rate']], 
                    use_container_width=True)
    
    with col2:
        st.subheader("🎯 Top Performer")
        top_brand = competitor_df.loc[competitor_df['sentiment_score'].idxmax()]
        
        st.markdown(f"""
        <div class="brand-card">
            <h3>{top_brand['brand']}</h3>
            <p><strong>Sentiment Score:</strong> {top_brand['sentiment_score']:.3f}</p>
            <p><strong>Market Share:</strong> {top_brand['market_share']:.1f}%</p>
            <p><strong>Mentions:</strong> {top_brand['mention_volume']:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Keyword analysis
    st.subheader("🔤 Keyword Association Analysis")
    
    keywords_data = []
    for _, row in competitor_df.iterrows():
        for keyword in row['top_keywords']:
            keywords_data.append({
                'brand': row['brand'],
                'keyword': keyword,
                'frequency': random.randint(50, 300)
            })
    
    keywords_df = pd.DataFrame(keywords_data)
    pivot_keywords = keywords_df.pivot(index='keyword', columns='brand', values='frequency').fillna(0)
    
    fig = px.imshow(pivot_keywords.values, 
                   x=pivot_keywords.columns, 
                   y=pivot_keywords.index,
                   title="Brand-Keyword Association Heatmap",
                   color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

elif selected_page == "Topic Analysis":
    st.markdown("## 🏷️ Topic Analysis Dashboard")
    
    # Topic overview
    filtered_topics = topic_df[topic_df['brand'].isin(selected_brands)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Topic Mention Volume")
        topic_volume = filtered_topics.groupby('topic')['mention_count'].sum().reset_index()
        topic_volume = topic_volume.sort_values('mention_count', ascending=False)
        
        fig = px.pie(topic_volume, values='mention_count', names='topic',
                    title="Distribution of Topic Mentions")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("😊 Topic Sentiment Scores")
        topic_sentiment = filtered_topics.groupby('topic')['sentiment_score'].mean().reset_index()
        topic_sentiment = topic_sentiment.sort_values('sentiment_score', ascending=True)
        
        fig = px.bar(topic_sentiment, x='sentiment_score', y='topic',
                    orientation='h', title="Average Sentiment by Topic",
                    color='sentiment_score', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed topic analysis
    st.subheader("🔍 Detailed Topic Breakdown")
    
    selected_topic = st.selectbox("Select Topic for Detailed Analysis", filtered_topics['topic'].unique())
    
    topic_detail = filtered_topics[filtered_topics['topic'] == selected_topic]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_mentions = topic_detail['mention_count'].sum()
        st.metric("Total Mentions", f"{total_mentions:,}")
    
    with col2:
        avg_sentiment = topic_detail['sentiment_score'].mean()
        st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
    
    with col3:
        trend_up = len(topic_detail[topic_detail['trend'] == 'up'])
        st.metric("Brands Trending Up", trend_up)
    
    # Brand comparison for selected topic
    fig = px.bar(topic_detail, x='brand', y='mention_count',
                color='sentiment_score', title=f"{selected_topic} - Brand Comparison",
                color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)

elif selected_page == "Real-time Monitoring":
    st.markdown("## ⚡ Real-time Social Media Monitoring")
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mentions_now = st.metric("Mentions (Last Hour)", "342", "23 new")
    with col2:
        sentiment_now = st.metric("Current Sentiment", "74.2%", "1.3%")
    with col3:
        engagement_now = st.metric("Engagement Rate", "5.8%", "0.4%")
    with col4:
        alerts_now = st.metric("Active Alerts", "3", "1 new")
    
    # Live feed simulation
    st.subheader("📱 Live Social Media Feed")
    
    # Create placeholder for live updates
    placeholder = st.empty()
    
    if st.button("Start Real-time Monitoring"):
        for i in range(10):
            with placeholder.container():
                # Simulate real-time updates
                sample_posts = [
                    {"platform": "Twitter", "brand": "Tesla", "text": "Just test drove the new Model 3. Amazing acceleration!", "sentiment": "Positive", "timestamp": datetime.now() - timedelta(minutes=i*2)},
                    {"platform": "Instagram", "brand": "BMW", "text": "BMW's design team really outdid themselves with the new X5", "sentiment": "Positive", "timestamp": datetime.now() - timedelta(minutes=i*2+1)},
                    {"platform": "Facebook", "brand": "Mercedes", "text": "Mercedes service center experience was disappointing", "sentiment": "Negative", "timestamp": datetime.now() - timedelta(minutes=i*2+2)},
                ]
                
                for post in sample_posts[:3]:
                    if post['brand'] in selected_brands:
                        sentiment_color = "#2ecc71" if post['sentiment'] == "Positive" else "#e74c3c"
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; padding: 1rem; margin: 0.5rem 0; border-radius: 5px;">
                            <div style="display: flex; justify-content: space-between;">
                                <strong>{post['platform']} | {post['brand']}</strong>
                                <span style="color: {sentiment_color};">{post['sentiment']}</span>
                            </div>
                            <p>{post['text']}</p>
                            <small>{post['timestamp'].strftime('%H:%M:%S')}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            time.sleep(1)
    
    # Alert management
    st.subheader("🚨 Alert Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Active Alerts")
        alerts = [
            {"brand": "Tesla", "type": "Sentiment Drop", "threshold": "-15%", "status": "Active"},
            {"brand": "BMW", "type": "Mention Spike", "threshold": "+200%", "status": "Investigating"},
            {"brand": "Mercedes", "type": "Negative Keywords", "threshold": "High", "status": "Resolved"}
        ]
        
        for alert in alerts:
            status_color = {"Active": "#e74c3c", "Investigating": "#f39c12", "Resolved": "#2ecc71"}[alert['status']]
            st.markdown(f"""
            <div style="border-left: 4px solid {status_color}; padding: 1rem; margin: 1rem 0; background: #f8f9fa;">
                <strong>{alert['brand']}</strong> - {alert['type']}<br>
                Threshold: {alert['threshold']} | Status: <span style="color: {status_color};">{alert['status']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Configure New Alert")
        alert_brand = st.selectbox("Brand", selected_brands)
        alert_type = st.selectbox("Alert Type", ["Sentiment Drop", "Mention Spike", "Negative Keywords", "Competitor Mention"])
        alert_threshold = st.slider("Threshold", -50, 50, 10)
        
        if st.button("Create Alert"):
            st.success(f"Alert created for {alert_brand} - {alert_type}")

elif selected_page == "Reports & Insights":
    st.markdown("## 📊 Reports & Strategic Insights")
    
    # Report generation
    st.subheader("📈 Automated Report Generation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        report_type = st.selectbox("Report Type", [
            "Weekly Summary",
            "Competitive Analysis",
            "Brand Health Check",
            "Market Trends",
            "Crisis Management"
        ])
    
    with col2:
        report_brands = st.multiselect("Include Brands", sentiment_df['brand'].unique(), default=selected_brands)
    
    with col3:
        report_format = st.selectbox("Format", ["PDF", "PowerPoint", "Excel", "Interactive Dashboard"])
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            time.sleep(2)
            st.success("Report generated successfully!")
            st.download_button(
                label="Download Report",
                data="Sample report content would be here",
                file_name=f"AutoSense_{report_type.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
    
    # Strategic insights
    st.subheader("🧠 AI-Generated Strategic Insights")
    
    insights = [
        {
            "title": "Market Opportunity: Electric Vehicle Sentiment",
            "content": "Consumer sentiment towards electric vehicles has increased 34% over the past quarter. Tesla leads with 78% positive sentiment, while traditional automakers show growing EV interest. Recommendation: Accelerate EV marketing campaigns during this positive sentiment wave.",
            "priority": "High",
            "impact": "Revenue Growth"
        },
        {
            "title": "Competitive Threat: Luxury Segment Disruption",
            "content": "BMW and Mercedes face increasing price-value concerns, with 23% of mentions citing 'overpriced' keywords. Meanwhile, Genesis and Lexus gain positive sentiment in luxury discussions. Recommendation: Review pricing strategy and emphasize value proposition.",
            "priority": "Medium",
            "impact": "Market Share"
        },
        {
            "title": "Customer Experience Gap: Service Quality",
            "content": "Service-related negative sentiment increased 18% across all monitored brands. Tesla shows highest service dissatisfaction (31% negative), while Lexus maintains positive service sentiment (67% positive). Recommendation: Implement service quality improvement programs.",
            "priority": "High",
            "impact": "Customer Retention"
        }
    ]
    
    for insight in insights:
        priority_color = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}[insight['priority']]
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; padding: 1.5rem; margin: 1rem 0; border-radius: 8px; background: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h4 style="margin: 0;">{insight['title']}</h4>
                <div>
                    <span style="background: {priority_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem;">{insight['priority']}</span>
                    <span style="background: #007bff; color: white; padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem; margin-left: 0.5rem;">{insight['impact']}</span>
                </div>
            </div>
            <p>{insight['content']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance summary
    st.subheader("📊 Performance Summary")
    
    summary_data = {
        'Metric': ['Total Mentions Analyzed', 'Brands Monitored', 'Social Platforms', 'Languages Supported', 'Accuracy Rate', 'Response Time'],
        'Value': ['2.3M', '50+', '12', '25', '94.7%', '<2 sec'],
        'Change': ['+15%', '+8', '+2', '+5', '+2.1%', '-15%']
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>AutoSense</strong> - Revolutionizing Automotive Market Intelligence</p>
    <p>Powered by Advanced AI & Real-time Social Analytics | Version 2.1.0</p>
</div>
""", unsafe_allow_html=True)