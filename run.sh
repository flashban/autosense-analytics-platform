echo "🚗 Starting AutoSense Analytics Platform..."
echo "📊 Installing dependencies..."
pip install -r requirements.txt
echo "🚀 Launching Streamlit app..."
streamlit run app.py --server.address 0.0.0.0 --server.port 8501