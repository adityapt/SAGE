# SAGE - AI MMM Copilot

AI-powered Marketing Mix Modeling assistant with natural language interface.

## Features

- 🤖 Natural language queries for MMM analysis
- 📈 Automatic response curve fitting (Hill curves)
- 💰 Budget optimization with constraints
- 🧠 Domain knowledge (RAG with ChromaDB)
- 📊 Interactive Plotly visualizations
- 🔄 Feedback system with alternative answers

## Quick Start

### Streamlit Cloud (Easiest)

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in with GitHub
3. Click "New app"
4. Select this repository
5. Main file: `app.py`
6. Deploy!

### Local Development

```bash
# Clone
git clone https://github.com/adityapt/SAGE.git
cd SAGE

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

## Usage

1. **Enter OpenAI API Key** (sidebar)
2. **Upload CSV** or use sample data
3. **Ask questions!**

Example queries:
- "What is TV's ROI?"
- "Allocate 100M across channels"
- "Show me response curves"

## Data Format

CSV with columns: `date`, `channel`, `spend`, `impressions`, `predicted`

See `data/sample_template.csv` for example.

## License

MIT

