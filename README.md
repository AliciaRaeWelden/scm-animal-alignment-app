# 🧠 SCM Animal Alignment App

This is a lightweight Streamlit app for testing language model outputs against semantic alignment metrics related to **animal framing** — inspired by real-world ethical disambiguation challenges.

You can enter a prompt, choose a Hugging Face model, and instantly see how the model's completion aligns with:

- 🌱 **Vegan empathy framing** (sentience, individuality, moral concern)
- 🏭 **Commodity framing** (animals as products or labor)

---

## 🚀 Try the Live App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/AliciaRaeWelden/scm-animal-alignment-app/main/scm_app.py)

---

## 📦 How to Run It Locally

```bash
git clone git@github.com:AliciaRaeWelden/scm-animal-alignment-app.git
cd scm-animal-alignment-app
pip install -r requirements.txt
streamlit run scm_app.py
```

---

## 🛠 Models Supported

- `mistralai/Mistral-7B-Instruct-v0.1`
- `tiiuae/falcon-7b-instruct`
- `google/flan-t5-large`

These run through the [Hugging Face Inference API](https://huggingface.co/inference-api) — no login required for public models.

---

## 📊 About SCM Scoring

We compute semantic alignment using cosine similarity between the model's response and two concept centroids:

- A **vegan empathy centroid**
- A **commodity framing centroid**

You’ll see the difference visualized as:

```
Δ = (Vegan Similarity) − (Commodity Similarity)
```

---

## ✨ Credit

Built by [AliciaRaeWelden](https://github.com/AliciaRaeWelden)  
Powered by [Streamlit](https://streamlit.io) and [Hugging Face](https://huggingface.co)
