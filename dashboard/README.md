# NEXUS Alpha v3 - Pro dashboard (A3b)

Files
- `app_pro.py`  the dashboard (dark / light, candlestick background, plain-English read-outs, candles joined to the Monte Carlo cone, Evidence / Models / Guide pages)
- `.streamlit/config.toml`  theme base and server options
- `NEXUS_Alpha_v3_Dashboard_Launcher.ipynb`  open in Colab, run the 2 cells, open the link

It needs the A3b dashboard bundle (bundle.json, nav3_core.py, nav3_engine.py, screen_reader.py, models) that the main notebook writes to
`MyDrive/antigravity_workspace/nexus_alpha_v3/artifacts_full_research/app/`. `app_pro.py` sits next to them; the original `app.py` is untouched.
The model paths inside bundle.json are Colab paths, so the app runs in Colab, not directly on Windows.
