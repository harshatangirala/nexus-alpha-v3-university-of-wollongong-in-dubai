# NEXUS Alpha v3: lite dashboard

Public, always-on version of the dashboard for Streamlit Community Cloud: <https://nexus-alpha-v3-lite.streamlit.app>

The full dashboard (`dashboard/app_pro.py`) loads 13 neural networks (2.9 GB of weights, about 5 GB of memory). That is more than free hosting allows, so it runs on the author's machine and is online during demonstrations: <https://buffalo-deposits-graph-legends.trycloudflare.com>

## What the lite version shows

- **Example analyses.** Real output of the full frozen models for NVDA, V (Visa), F (Ford) and BAC (Bank of America). Each was computed from three TradingView screenshots (1W, 1D, 1H) taken on 24 November 2026, with the full dashboard's own code path (`dashboard/screen_engine.analyse_screens`: screenshot reader, 7 chart networks, calibration, Monte Carlo with 4,000 paths). The results are saved in `examples/*.json`, and the three charts the networks saw are in `examples/*_model_input.png`. No filing text was added, so these outlooks come from the charts only.
- **Ask AI.** The same grounded Gemini assistant as the full dashboard, with the four examples as context.
- **Architecture, Models and Guide.** The same pages as the full dashboard.

The look, the result cards, the study facts and the assistant are read from `dashboard/app_pro.py` when the app starts. Only constants and helper functions are taken, nothing that loads a model, so the two dashboards cannot drift apart. No PyTorch and no weights are needed.

## Deploy on Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io> with the GitHub account that owns this repository.
2. Click **Create app**, choose **Deploy a public app from GitHub**, then pick this repository, branch `main` and main file path `lite/streamlit_app.py`.
3. Under **App URL**, type `nexus-alpha-v3-lite` so the address is `https://nexus-alpha-v3-lite.streamlit.app`.
4. Optional, for Ask AI without asking visitors for a key: open **Advanced settings**, then **Secrets**, and add `GEMINI_API_KEY = "your-key"`. The key stays in Streamlit's secret store and is never committed.
5. Click **Deploy**. The app is public by default.

## Run locally

```bash
pip install -r lite/requirements.txt
streamlit run lite/streamlit_app.py
```
