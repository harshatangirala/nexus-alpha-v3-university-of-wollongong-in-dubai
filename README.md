# NEXUS Alpha v3

**University of Wollongong in Dubai · Master of Applied AI · Final project**

| | Link |
|---|---|
| Lite dashboard (public, always on) | <https://nexus-alpha-v3-lite.streamlit.app> |
| Full dashboard (13 networks, runs on the author's machine during demonstrations) | <https://buffalo-deposits-graph-legends.trycloudflare.com> |
| CV dataset (price bars the charts are drawn from) | <https://drive.google.com/file/d/1Y7wvfpJcjnmXLP79hHTih-jj5pt-muu-/view> |
| NLP dataset (EDGAR filing text and metadata) | <https://drive.google.com/drive/folders/1nczPvpd_wkMgF5N7uFBfAQMp-Ja96TRI> |
| All project data | <https://drive.google.com/drive/folders/13rokT8OX1MgZQpzTRLeyvDq-FVdgraSh> |
| Group deliverables (CV report, NLP report, IEEE paper) | [`deliverables/`](deliverables/) |

The full dashboard loads 13 neural networks (2.9 GB of weights, about 5 GB of memory), which is too heavy for free hosting. The **lite version** on Streamlit Community Cloud needs no model weights: it shows real output of the full models for four stocks, the architecture, the model routing and the Ask AI assistant. See [`lite/README.md`](lite/README.md).

A multimodal study of whether **chart images** and **SEC filing text**, read by 13 neural networks, add predictive information about US stocks beyond what simple price statistics already give. Seven vision networks read weekly, daily and hourly candlestick charts; six language models read 10-K, 10-Q and 8-K filings. The system estimates, at 5, 10, 20 and 40 trading sessions:

| Target | Question | Metric |
|---|---|---|
| Direction | DOWN / SIDEWAYS / UP (band = 0.5 x the stock's own volatility x sqrt(horizon)) | macro-F1 against a shuffled-label null |
| Volatility | how jumpy will the price be | R-squared against predicting the average |
| Drawdown | will the price fall more than 10 % below today's close | AUC (0.5 = guessing) |

> Educational research. Nothing here is financial advice.

## Design in one paragraph

50 large US stocks (10 in each of 5 sectors), prices Oct 2016 to Oct 2026, 10,269 EDGAR filings. Time-based splits with a 40-session gap: train 2016 to 2020 (51,250 rows), early stopping 2021, model selection 2022-2023, and a **locked test 2024-01 to 2026-08 (6,550 rows)** that was sealed with a hashed freeze and opened once. Seven hypotheses were pre-registered (H1 to H7); every later analysis (H1c onward) is labelled **post-hoc**. Holm correction and bootstrap confidence intervals are used throughout, and models are compared against strong numbers-only benchmarks, not just chance.

## What was found (locked test, 2024 to 2026)

| | vs no-skill benchmark | vs strong numbers-only baseline |
|---|---|---|
| **Computer vision** | supported in 6 of 12 tests | better in 4 (all direction), worse in 8 |
| **NLP** | supported in 4 of 12 (all drawdown) | better in 1, level in 7, worse in 4 |
| **CV + NLP combined** | beats the better single modality in 6 of 12 | better in 4 (all direction), worse in 7 |

* A **small but statistically supported direction signal** from the chart networks (about +0.03 macro-F1 at 5 and 10 sessions). Plain accuracy is about the same as always guessing SIDEWAYS.
* On **volatility and drawdown, simple numbers (HAR, momentum, technical-indicator LightGBM) beat the charts and the text.** Volatility R-squared is negative at long horizons: each chart is scaled to its own high and low, which hides the absolute volatility level.
* Most of the **text signal is company and calendar shortcuts**. Filing days do move prices about 2.1 times as much as ordinary days, and the words help explain how big the reaction is (+0.043 R-squared beyond company, post-hoc).
* The dashboard recipe beats the no-skill benchmark on all 4 direction tasks, all 4 drawdown tasks and 1 of 4 volatility tasks.

Limitations: 50 large surviving US stocks, one decade with a regime change, one training seed per network, no trading costs, and a post-hoc analysis phase after the test was opened.

## Repository layout

```
deliverables/ group submissions: CV_Report, NLP_Report and IEEE_Paper (LaTeX source for
              Overleaf, figures, and a PDF of each), plus the figure and PDF build scripts
lite/         the lite Streamlit dashboard for public hosting (no model weights needed)
reports/      the three final PDF reports (Computer Vision, NLP, Multimodal)
notebook/     NEXUS_Alpha_v3_FINAL.ipynb (the full Colab run), notebook_src/ (the readable
              source it is built from, s00 to s26), build tools, history/ (earlier versions)
results/      tables/ (every result table from the final run) and figures/
dashboard/    the Streamlit dashboard: app, engine, screenshot reader, architecture and
              flowchart views, bundle.json (the frozen recipe), Dockerfile, requirements
docs/         project brief, design and run guides, faculty presentation, study guide
```

## The dashboard

`dashboard/app_pro.py` is a Streamlit app that applies the frozen recipe:

* **Analyse**: type a ticker or paste TradingView screenshots (1W, 1D, 1H). With screenshots, candles, prices and volume are read from the pictures (OCR of the price axis plus a pixel reader, median price error 0.08 % on 66 real screenshots), then redrawn in the training format and passed to the networks. Shows calls, probabilities, volatility, drawdown risk, a 20-session Monte Carlo price cone, descriptive candlestick patterns, and an interactive flowchart of which network did what.
* **Ask AI**: a Google Gemini chat that explains the result and the study (needs a free Gemini key, see below).
* **Architecture**: interactive pipeline from data collection to prediction, with hover explanations.
* **Models** and **Guide**.

### Run it locally

```bash
cd dashboard
pip install -r requirements.txt
# put the model files next to the app in dashboard/models/  (see "Model weights")
streamlit run app_pro.py
```

For the Ask AI tab, set `GEMINI_API_KEY` (get a free key at aistudio.google.com/apikey) as an environment variable or in `.streamlit/secrets.toml`. **Never commit a key.**

### Model weights

The 13 fine-tuned networks are about 2.9 GB, over GitHub's file limit, so they are **not stored here**. They are produced by running the notebook (`notebook/NEXUS_Alpha_v3_FINAL.ipynb`) on a GPU runtime. The file names the dashboard expects are listed in `dashboard/bundle.json` (they go in `dashboard/models/`).

## Reproducing the research

Open `notebook/NEXUS_Alpha_v3_FINAL.ipynb` in Google Colab (a GPU runtime), mount Google Drive and run all cells (about 4 hours end to end, of which the SEC download is about 2 hours). `docs/RUN_GUIDE.md` has the details. The readable source of every section is in `notebook/notebook_src/`.

## Data

The data are too large for GitHub and are shared on public Google Drive links (table at the top):

* **CV dataset** (`nav3_bars_66.zip`, 112 MB): the cleaned price bars from which the notebook's renderer redraws every chart image byte for byte.
* **NLP dataset** (folders `submissions/` and `texts/`): the EDGAR submission records and the extracted filing text used by the language models.

SEC filings are public records, downloaded under the SEC fair-access policy. The price bars start from a public 1-minute dataset that has no declared licence; they are shared here for non-commercial academic reproduction only. The data-collection and labelling code is in `notebook/notebook_src/s03_data.pct`, `s04_labels.pct` and `s13_sec_text.pct`.

## Deliverables

| Document | Source (Overleaf) | PDF | Body words |
|---|---|---|---|
| Computer Vision report (MAAI 901) | `deliverables/CV_Report/CV_Report.tex` | `deliverables/CV_Report/CV_Report.pdf` | about 2,800 (limit 2,500 to 3,000) |
| NLP report (MAAI 902, section 5.3.3) | `deliverables/NLP_Report/NLP_Report.tex` | `deliverables/NLP_Report/NLP_Report.pdf` | about 3,400 (limit 3,000 to 4,000) |
| Unified research paper (IEEE conference format) | `deliverables/IEEE_Paper/IEEE_Paper.tex` | `deliverables/IEEE_Paper/IEEE_Paper.pdf` | about 5,100 |

To compile on Overleaf, upload one document folder (the `.tex` file and its `figures/` folder) and compile with pdfLaTeX. See `deliverables/README.md`.

## Author

Krishna Harsha Tangirala, Master of Applied AI, University of Wollongong in Dubai.
