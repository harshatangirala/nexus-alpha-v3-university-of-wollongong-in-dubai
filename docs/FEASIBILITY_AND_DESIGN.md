# Feasibility check and design (2026-11-21)

## 1. Data availability for the 50 stocks in the brief

Checked against the TriFrame frozen tier-A store (`Downloads\us_stocks_ohlcv\final\`, 979 verified stocks, 1-week / 1-day / 1-hour bars, frozen 2026-10-11, data through 2026-10-09; hashes equal NEXUS V2's manifest).

| Sector | In the store (kept) | Not in the store → replacement (same sector) |
|---|---|---|
| Information Technology | AAPL MSFT NVDA INTC CSCO ORCL QCOM AMD | AVGO → **TXN**, CRM → **ADBE** |
| Financials | JPM BAC WFC C GS MS V MA BRK.B SCHW | none |
| Health Care | UNH JNJ LLY PFE MRK ABT TMO AMGN GILD CVS | none |
| Consumer Discretionary | AMZN HD MCD NKE SBUX F LOW TJX ROST | BKNG → **MAR** |
| Communication Services | NFLX DIS T VZ TTWO LYV | GOOGL → **GOOG** (same company), CMCSA → **META**, TMUS → **OMC**, EA → **NYT** |

Why: the replaced names have no verified 10-year **hourly** history (Yahoo serves only 730 days of hourly bars). Every kept stock has bars from 2001 (V 2008, MA 2006, META 2012, LYV/SCHW/T/MS 2005-06), far more than the 2014-06 warm-up the charts need. Reserves bundled in the zip (used automatically, by a fixed rule, only if a stock fails the SEC check in section 3): IT ADI MU IBM INTU; Financials AXP PNC USB BLK; Health Care MDT BMY DHR ISRG; Consumer Discretionary EBAY YUM ORLY; Communication Services IMAX.

SEC text: EDGAR has 10-K, 10-Q and 8-K filings for all 50; the notebook downloads them with point-in-time acceptance timestamps. CIK history is pinned where a company changed filer (Disney 1744489 since 2019, 1001039 before; BlackRock reserve 2012383 / 1364742).

## 2. Model availability

| Brief | Used | Note |
|---|---|---|
| Custom CNN (JKX) | JKX-style CNN, from scratch | widened to 4 blocks for 224 px |
| ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16, Swin-T | timm ImageNet weights | fine-tuned and frozen |
| YOLOv8 "chart-pattern weights" | YOLOv8s-cls backbone (fine-tuned) + `foduucom/stockmarket-pattern-detection-yolov8` detector (zero-shot features) | a detector predicts boxes, not returns |
| FinBERT | `ProsusAI/finbert` | |
| SEC-BERT | `nlpaueb/sec-bert-base` (pre-trained on 260,773 10-Ks, 1993-2019) | |
| RoBERTa-large (Financial) | `FacebookAI/roberta-large` | no established financial RoBERTa-large exists |
| Longformer | `allenai/longformer-base-4096` | |
| DeBERTa-v3-large | `microsoft/deberta-v3-large` | |
| FinGPT | `FinGPT/fingpt-mt_llama2-7b_lora` on Llama-2-7B, 4-bit, frozen (+ optional QLoRA) | Llama-2 is gated |
| TF-IDF + Ridge / Logistic | scikit-learn | baseline |

## 3. Design decisions (made before any result)

* Decisions 2016-10-10 → 2026-08 (10 years); train 2016-10 → 2020-12, inner 2021, validation 2022-23, locked test 2024-01 → 2026-08; 40-session purge.
* Targets per horizon: 3-class direction with a ±0.5 σ√h band; log realised volatility from hourly bars; >10 % drawdown.
* All 12 tasks from one multi-task network per model; identical training recipe for all CV models and for all NLP encoders; calibration on 2021 only.
* Selection on validation only; ensembles are equal-weight; fusion weights from a 5-point grid on validation; everything frozen and hashed before the test block opens.
* Test period after every language model's pre-training cut-off (look-ahead-safe text).
* Dashboard = the frozen `deploy_recipe`, parity-tested against the notebook; screenshots are read (OCR) and redrawn in the training format; no direction call where test skill is not significant.

## 4. Expected outcome (stated in advance)

From TriFrame, NEXUS V1 and the SEC study on overlapping data: volatility is predictable (mostly persistence), drawdown partly, direction close to chance; images and text add little over simple numbers. The study is built so that a clear null is a defensible result.
