import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from pathlib import Path
import io

# Try to load .env file for local development
try:
    from dotenv import load_dotenv
    env_paths = [
        Path(__file__).parent / '.env',
        Path(__file__).parent.parent / 'Journals/llm-copilot/.env'
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

from llm_copilot import MMMCopilot

st.set_page_config(
    page_title="SAGE - Strategic AI Marketing Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: #FFFFFF;
        padding: 0;
    }
    
    .block-container {
        padding-top: 2rem !important;
        max-width: 1400px !important;
    }
    
    .stChatMessage {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: none;
        margin-bottom: 1.25rem;
    }
    
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        color: white;
        border: none;
    }
    
    .stChatInputContainer {
        background: white !important;
        border-top: 2px solid #E2E8F0 !important;
        padding: 2rem 1rem !important;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.08) !important;
    }
    
    input {
        border-radius: 8px !important;
        border: 2px solid #E2E8F0 !important;
        font-size: 1rem !important;
        padding: 0.875rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    
    input:focus {
        border-color: #667EEA !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    h1 {
        display: none;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        color: white;
        text-align: center;
        margin: -2rem -1rem 2.5rem -1rem;
        padding: 2.5rem 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .hero-title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .hero-icon {
        width: 48px;
        height: 48px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.9;
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.5;
    }
    
    .welcome-card {
        background: white;
        border-radius: 12px;
        padding: 2.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: none;
        margin-bottom: 1.5rem;
    }
    
    .welcome-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 0.75rem;
    }
    
    .welcome-subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    .example-query {
        background: #F8FAFC;
        border-left: 3px solid #667EEA;
        padding: 0.875rem 1rem;
        margin: 0.625rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.95rem;
        color: #334155;
        transition: all 0.2s ease;
        cursor: default;
    }
    
    .example-query:hover {
        background: #EEF2FF;
        border-left-color: #4F46E5;
    }
    
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border: 1px solid #D1D5DB;
        background: white;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        border-color: #3B82F6;
        color: #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .sidebar .stTextInput>div>div>input {
        border: 2px solid #E5E7EB;
    }
    
    .stSpinner {
        color: #667EEA;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR: Configuration
# ============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key Input
    st.markdown("### 🔑 OpenAI API Key")
    
    # Check if API key exists in environment
    env_api_key = os.getenv('OPENAI_API_KEY', '')
    
    if env_api_key:
        st.success("✅ API Key loaded from environment")
        api_key_input = env_api_key
        show_key_input = st.checkbox("Enter different API key", value=False)
        if show_key_input:
            api_key_input = st.text_input(
                "API Key",
                type="password",
                value="",
                help="Enter your OpenAI API key (starts with sk-...)"
            )
    else:
        api_key_input = st.text_input(
            "API Key",
            type="password",
            value="",
            placeholder="sk-...",
            help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys"
        )
        
        if not api_key_input:
            st.warning("⚠️ Please enter your OpenAI API key to continue")
            st.markdown("""
            **How to get an API key:**
            1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Sign in or create an account
            3. Click "Create new secret key"
            4. Copy and paste it here
            """)
    
    st.markdown("---")
    
    # Data Upload Section
    st.markdown("### 📊 Data Source")
    
    data_source = st.radio(
        "Choose data source:",
        ["Upload CSV", "Use Sample Data"],
        help="Upload your own MMM data or use sample data for testing"
    )
    
    uploaded_data = None
    
    if data_source == "Upload CSV":
        st.markdown("#### Upload Your MMM Data")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload a CSV file with columns: date, channel, spend, impressions, predicted"
        )
        
        if uploaded_file:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_cols = ['date', 'channel', 'spend', 'impressions', 'predicted']
                missing_cols = [col for col in required_cols if col not in uploaded_data.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                    st.markdown("**Required columns:**")
                    for col in required_cols:
                        st.markdown(f"- `{col}`")
                    uploaded_data = None
                else:
                    # Convert date column
                    uploaded_data['date'] = pd.to_datetime(uploaded_data['date'])
                    
                    st.success(f"✅ Data loaded: {len(uploaded_data)} rows")
                    st.markdown(f"**Channels:** {', '.join(uploaded_data['channel'].unique())}")
                    st.markdown(f"**Date range:** {uploaded_data['date'].min().date()} to {uploaded_data['date'].max().date()}")
                    
                    # Show data preview
                    with st.expander("📋 Preview Data"):
                        st.dataframe(uploaded_data.head(10), use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                uploaded_data = None
        
        else:
            st.info("👆 Upload a CSV file to get started")
            
            # Show CSV template
            with st.expander("📄 Download CSV Template"):
                st.markdown("**Template format:**")
                template_data = pd.DataFrame({
                    'date': ['2024-01-01', '2024-01-08', '2024-01-15'],
                    'channel': ['TV', 'TV', 'Search'],
                    'spend': [50000, 55000, 30000],
                    'impressions': [1000000, 1100000, 500000],
                    'predicted': [67000, 72000, 45000]
                })
                st.dataframe(template_data, use_container_width=True)
                
                # Download button
                csv_template = template_data.to_csv(index=False)
                st.download_button(
                    label="💾 Download Template",
                    data=csv_template,
                    file_name="mmm_data_template.csv",
                    mime="text/csv"
                )
    
    else:  # Use Sample Data
        st.info("📊 Using synthetic sample data with 5 channels and 52 weeks")
        with st.expander("ℹ️ Sample Data Details"):
            st.markdown("""
            **Channels:** TV, Search, Social, Display, Radio
            
            **Metrics:**
            - Weekly spend (with realistic variation)
            - Impressions
            - Predicted sales (using Hill curves)
            
            **Features:**
            - Realistic saturation effects
            - Diminishing returns
            - Channel-specific parameters
            """)

    st.markdown("---")
    
    # Status indicator
    if api_key_input and (uploaded_data is not None or data_source == "Use Sample Data"):
        st.success("✅ Ready to analyze!")
    else:
        missing = []
        if not api_key_input:
            missing.append("API Key")
        if data_source == "Upload CSV" and uploaded_data is None:
            missing.append("Data")
        st.warning(f"⚠️ Missing: {', '.join(missing)}")

# ============================================================================
# MAIN APP
# ============================================================================

# Initialize copilot
def init_copilot(api_key, data=None):
    """Initialize MMM copilot with provided API key and data"""
    
    if not api_key:
        return None
    
    # Use uploaded data or generate sample data
    if data is None:
        # Generate sample data
        np.random.seed(42)
        channels = ['TV', 'Search', 'Social', 'Display', 'Radio']
        weeks = 52
        
        def hill_curve(spend, top, saturation, slope):
            return top * (spend ** slope) / (saturation ** slope + spend ** slope)
        
        data_list = []
        saturation_params = {
            'TV': {'top': 500000, 'saturation': 150000, 'slope': 0.8, 'base_spend': 100000},
            'Search': {'top': 400000, 'saturation': 80000, 'slope': 0.9, 'base_spend': 50000},
            'Social': {'top': 350000, 'saturation': 60000, 'slope': 1.0, 'base_spend': 40000},
            'Display': {'top': 250000, 'saturation': 50000, 'slope': 0.85, 'base_spend': 30000},
            'Radio': {'top': 180000, 'saturation': 40000, 'slope': 0.75, 'base_spend': 25000}
        }
        
        for channel in channels:
            params = saturation_params[channel]
            for week in range(weeks):
                date = pd.Timestamp('2024-01-01') + pd.Timedelta(weeks=week)
                spend = params['base_spend'] * (1 + np.random.uniform(-0.3, 0.3))
                impressions = spend * 50 * (1 + np.random.uniform(-0.2, 0.2))
                base_response = hill_curve(spend, params['top'], params['saturation'], params['slope'])
                predicted = base_response * (1 + np.random.uniform(-0.1, 0.1))
                
                data_list.append({
                    'date': date,
                    'channel': channel,
                    'spend': spend,
                    'impressions': impressions,
                    'predicted': predicted
                })
        
        data = pd.DataFrame(data_list)
    
    try:
        copilot = MMMCopilot(data=data, api_key=api_key)
        return copilot
    except Exception as e:
        st.error(f"❌ Error initializing copilot: {str(e)}")
        return None

# Check if configuration is complete
config_complete = api_key_input and (uploaded_data is not None or data_source == "Use Sample Data")

if not config_complete:
    # Show beautiful hero header
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title-container">
            <span class="hero-title">📊 SAGE</span>
        </div>
        <div class="hero-subtitle">Strategic AI-Guided Explorer for Marketing Performance</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show welcome card with example queries
    with st.container():
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 2.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 1.5rem;">
            <h2 style="font-size: 1.75rem; font-weight: 600; color: #1E293B; margin-bottom: 0.75rem;">Welcome to SAGE</h2>
            <p style="color: #64748B; font-size: 1.05rem; margin-bottom: 1.5rem; line-height: 1.6;">Your AI-powered marketing advisor. Ask strategic questions in plain English to optimize your media mix.</p>
            
            <div style="background: #F8FAFC; border-left: 3px solid #667EEA; padding: 0.875rem 1rem; margin: 0.625rem 0; border-radius: 0 4px 4px 0; color: #334155;">What's our best performing channel?</div>
            <div style="background: #F8FAFC; border-left: 3px solid #667EEA; padding: 0.875rem 1rem; margin: 0.625rem 0; border-radius: 0 4px 4px 0; color: #334155;">How should I allocate $50M across channels?</div>
            <div style="background: #F8FAFC; border-left: 3px solid #667EEA; padding: 0.875rem 1rem; margin: 0.625rem 0; border-radius: 0 4px 4px 0; color: #334155;">Show me TV saturation curves</div>
            <div style="background: #F8FAFC; border-left: 3px solid #667EEA; padding: 0.875rem 1rem; margin: 0.625rem 0; border-radius: 0 4px 4px 0; color: #334155;">Which channels should get more budget?</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👈 **Configure your settings in the sidebar to begin!**")
    
else:
    # Initialize session state
    if 'copilot' not in st.session_state or st.session_state.get('needs_reinit'):
        with st.spinner("🚀 Initializing MMM Copilot..."):
            data_to_use = uploaded_data if data_source == "Upload CSV" else None
            copilot = init_copilot(api_key_input, data_to_use)
            
            if copilot:
                st.session_state.copilot = copilot
                st.session_state.mmm_data = copilot.data
                st.session_state.messages = []
                st.session_state.feedback = {}
                st.session_state.needs_reinit = False
            else:
                st.error("❌ Failed to initialize copilot. Please check your configuration.")
                st.stop()
    
    # Display beautiful hero header
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title-container">
            <span class="hero-title">📊 SAGE</span>
        </div>
        <div class="hero-subtitle">Strategic AI-Guided Explorer for Marketing Performance</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show visualizations if available
            if message["role"] == "assistant" and "visualizations" in message:
                for viz_idx, viz in enumerate(message["visualizations"]):
                    st.plotly_chart(viz, use_container_width=True, key=f"viz_{idx}_{viz_idx}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your MMM data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get copilot response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                try:
                    response = st.session_state.copilot.query(prompt)
                    
                    answer = response.get("answer", "I encountered an issue processing your request.")
                    visualizations = response.get("visualizations", [])
                    
                    st.markdown(answer)
                    
                    for viz in visualizations:
                        st.plotly_chart(viz, use_container_width=True)
                    
                    # Store in session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "visualizations": visualizations
                    })
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

