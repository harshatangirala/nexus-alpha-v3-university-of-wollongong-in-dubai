So I want to make a project especially on colab notebook –

 

1\.       I want to do a multi modal project as part of my masters program related to computer vision plus natural language processing.  
2\.       The idea for computer vision is I would want to check which of the following 7 models are performing better at predicting the following when we show them just the candle sticks charts data for 50 stocks for 10 years timeline of 1 day charts and 1 week 1 hour charts –  
3\.       a. candle stick movement b. volatility c. drawdown chances of more than 10%  
4\.       the idea for NLP is I want to check which of the following 6 models are performing better at predicting the stock market movement and tell if if the stock will go upways or sideways or downways just by looking at the SEC filings of 8k, 10, and 10q data for 50 stock for 10 years timeline

 

Top 7 CV Models for Candlestick Chart Prediction

 

1\. Custom CNN – baseline from Jiang, Kelly & Xiu (2023) for chart images

2\. ResNet-50 – deeper CNN; Grad-CAM shows which candles mattered

3\. EfficientNet-B0 – lightweight and fast; tested on candlestick charts

4\. ConvNeXt-Tiny – modern CNN; adds novelty

5\. ViT-B/16 – Vision Transformer; tested on candlestick charts

6\. Swin Transformer-Tiny – hierarchical transformer; adds novelty

7\. YOLOv8 – detects candlestick patterns; ready-made chart-pattern weights available

 

Pretrained (fine-tune):

2\. ResNet-50

3\. EfficientNet-B0

4\. ConvNeXt-Tiny

5\. ViT-B/16

6\. Swin Transformer-Tiny

7\. YOLOv8

 

Top 7 NLP Models for Financial Statement Analysis & Stock Trading

1\. FinBERT – domain-specific BERT baseline (Araci, 2019); captures financial sentiment and tone in 10-K/10-Q filings

2\. SEC-BERT – pre-trained specifically on SEC EDGAR filings; handles complex accounting terminology and regulatory structure

3\. RoBERTa-large (Financial) – robustly optimized encoder; outperforms standard BERT on nuanced financial text classification

4\. Longformer (or BigBird) – sparse-attention transformer; reads entire multi-page financial statements (up to 4,096+ tokens) without truncation

5\. DeBERTa-v3-large – disentangled attention mechanism; adds novelty and high accuracy for subtle risk-factor extraction

6\. FinGPT – open-source financial LLM (LoRA fine-tuned); extracts structured ratios, summaries, and trading signals from statements

Pretrained (fine-tune / LoRA):

1\. FinBERT

2\. SEC-BERT

3\. RoBERTa-large

4\. Longformer

5\. DeBERTa-v3-large

6\. FinGPT

 

1. Base line models for computer vision: **Academic Gold-Standard Baseline:** **Custom Shallow CNN** (specifically the 3-to-4-block architecture from [Jiang, Kelly & Xiu (2023)](https://economics.yale.edu/research/re-imagining-price-trends)) and then**Pretrained Industry Baseline:** **ResNet-50** (or **EfficientNet-B0**). \[[1](https://arxiv.org/html/2408.08483v1), [2](https://www.astrj.com/pdf-208472-129234?filename=129234.pdf), [3](https://economics.yale.edu/research/re-imagining-price-trends)\]  
2. Base line models for NLP: **Sanity-Check Baseline: TF-IDF \+ Ridge / Logistic Regression**  
3. **Why:** In financial NLP (specifically Loughran-McDonald style bag-of-words), simple linear models on n-grams are notoriously hard to beat on 10-K/10-Q return prediction. If your deep learning model cannot beat TF-IDF out-of-sample, your pipeline has a data or signal issue.

\===================================================================

Here are the **50 U.S. stocks grouped by sector** with their company names and ticker symbols:

**Information Technology**

1. **Apple Inc.** (AAPL)  
2. **Microsoft Corporation** (MSFT)  
3. **NVIDIA Corporation** (NVDA)  
4. **Broadcom Inc.** (AVGO)  
5. **Intel Corporation** (INTC)  
6. **Cisco Systems, Inc.** (CSCO)  
7. **Oracle Corporation** (ORCL)  
8. **QUALCOMM Incorporated** (QCOM)  
9. **Advanced Micro Devices, Inc.** (AMD)  
10. **Salesforce, Inc.** (CRM)

**Financials**

1. **JPMorgan Chase & Co.** (JPM)  
2. **Bank of America Corporation** (BAC)  
3. **Wells Fargo & Company** (WFC)  
4. **Citigroup Inc.** (C)  
5. **The Goldman Sachs Group, Inc.** (GS)  
6. **Morgan Stanley** (MS)  
7. **Visa Inc.** (V)  
8. **Mastercard Incorporated** (MA)  
9. **Berkshire Hathaway Inc.** (BRK.B)  
10. **The Charles Schwab Corporation** (SCHW)

**Health Care**

1. **UnitedHealth Group Incorporated** (UNH)  
2. **Johnson & Johnson** (JNJ)  
3. **Eli Lilly and Company** (LLY)  
4. **Pfizer Inc.** (PFE)  
5. **Merck & Co., Inc.** (MRK)  
6. **Abbott Laboratories** (ABT)  
7. **Thermo Fisher Scientific Inc.** (TMO)  
8. **Amgen Inc.** (AMGN)  
9. **Gilead Sciences, Inc.** (GILD)  
10. **CVS Health Corporation** (CVS)

**Consumer Discretionary**

1. **Amazon.com, Inc.** (AMZN)  
2. **The Home Depot, Inc.** (HD)  
3. **McDonald's Corporation** (MCD)  
4. **NIKE, Inc.** (NKE)  
5. **Starbucks Corporation** (SBUX)  
6. **Ford Motor Company** (F)  
7. **Lowe's Companies, Inc.** (LOW)  
8. **The TJX Companies, Inc.** (TJX)  
9. **Booking Holdings Inc.** (BKNG)  
10. **Ross Stores, Inc.** (ROST)

**Communication Services**

1. **Alphabet Inc.** (GOOGL)  
2. **Netflix, Inc.** (NFLX)  
3. **The Walt Disney Company** (DIS)  
4. **AT\&T Inc.** (T)  
5. **Verizon Communications Inc.** (VZ)  
6. **Comcast Corporation** (CMCSA)  
7. **T-Mobile US, Inc.** (TMUS)  
8. **Electronic Arts Inc.** (EA)  
9. **Take-Two Interactive Software, Inc.** (TTWO)  
10. **Live Nation Entertainment, Inc.** (LYV)

 

Questions our Project Must Answer:

1\.       I want to check if computer vision is able to predict the movement of the stocks accurately for sideways, upways or downways for 5 days time period, 10 days time period , 20 days time period, 40 days time period  (or) is NLP able to predict the movement of the stocks accurately for sideways, upways or downways for 5 days time period, 10 days time period , 20 days time period, 40 days time period  (or) mixing both NLP plus computer vision is able to predict  the movement of sideways, upways or downways for 5 days time period, 10 days time period , 20 days time period, 40 days time period    
2\.       Which Computer vision model is performing the best on overall 50 stock provided and why ?  
3\.       Which computer vision model combination is performing better is it custom cnn \+ resnet or efficient net \+ rest net and so on and so forth  
4\.       Which individual computer vision model or which combination of computer vision model is able to predict the following right : a. movement of stock upways, side ways or downways…… b. volatility ……….. c. drawdown of 10%  
5\.       Which models are working efficiently on which of the 5 sectors and is there any particular neural network which is working better on an individual sector and not working good on other sectors and for each sector which neural network is working the best  
6\.       For NLP which report is more effecting the market movement is is 10k report or 8k report or 10q report  
7\.       For which particular sectors the 10k report and 8k report and 10q reports are moving the stock market more  
8\.       What are the particular things in these reports are moving the market  
9\.       Which sectors are being sensitive to the 10k report 8k report and 10 q report  
10\.  Is combining top 3 Computer vision networks and top 3 NLP networks is it adding better predictable edge ?

 

 

 

 

Product:

1\.       We need to make a streamlit dashboard where when ever I give screenshots of 4 stocks with two windows where I can input 1 week chart and 1 day chart and 1 hour chart  for each individual stock with a text bar where we provide recent text information summary of 10k, 8k, 10q summar the streamlit dashboard should provide us with the insights of will the stock will go upways, downways or sideways with Montecarlo simulation of price movement for next 20 days

\===================================================================

