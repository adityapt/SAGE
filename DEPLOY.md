# Deploy SAGE to GitHub

## ✅ Status: Ready to Deploy!

All files are committed and ready. Follow these steps:

---

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. **Repository name:** `SAGE`
3. **Description:** `AI-powered Marketing Mix Modeling Copilot`
4. **Visibility:** Public (so Streamlit Cloud can access it for free)
5. **Do NOT** check "Add README" (we already have one)
6. Click **"Create repository"**

---

## Step 2: Push Code

Copy and run these commands:

```bash
cd /Users/adityapu/Documents/GitHub/SAGE

# Add remote (replace 'adityapt' if your GitHub username is different)
git remote add origin https://github.com/adityapt/SAGE.git

# Push
git push -u origin main
```

---

## Step 3: Deploy to Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Sign in with your GitHub account
3. Click **"New app"**
4. **Repository:** `adityapt/SAGE`
5. **Branch:** `main`
6. **Main file path:** `app.py`
7. Click **"Deploy!"**

Wait ~5 minutes for deployment.

---

## Step 4: Access Your App

Your app will be live at:
```
https://adityapt-sage-app-main-[random].streamlit.app
```

---

## ✅ What's Included

- **app.py** - Streamlit UI with API key input & file upload
- **requirements.txt** - Installs llm-copilot from GitHub
- **data/sample_template.csv** - CSV template for users
- **.streamlit/config.toml** - Theme configuration
- **README.md** - Documentation

---

## 🎯 User Flow

1. User visits app URL
2. Enters OpenAI API key (sidebar)
3. Uploads CSV or uses sample data
4. Asks questions: "What is TV's ROI?"
5. Gets answer + visualizations

---

## That's it! 🚀

