# Run guide (Colab)

1. **Upload the data.** Put `nav3_bars_66.zip` (from `Downloads`) in `MyDrive/antigravity_workspace/nexus_alpha_v3/data/`. Do not unzip it.
2. **Secrets.** In Colab, key icon → add `SEC_USER_AGENT` = `Your Name your.email@domain` and `HF_TOKEN` = a Hugging Face read token. Switch *Notebook access* ON for each (it is off for every newly uploaded notebook).
3. **FinGPT (optional).** Accept the licence at https://huggingface.co/meta-llama/Llama-2-7b-hf with the same account as the token. Without it, FinGPT is recorded as *Not Run* and everything else continues. To skip it, set `RUN_FINGPT = False` in section 2.
4. **Runtime.** Runtime → Change runtime type → A100 GPU, High-RAM. Turn on background execution for the full run.
5. **QUICK_VERIFY.** Leave `RUN_MODE = "QUICK_VERIFY"` (section 2) and choose Runtime → Run all. About 1-1.5 h: 10 stocks (2 per sector), 1 epoch, short probes. Every output is stamped "not research results". It exercises every code path, including the freeze, the test opening, the dashboard and the three PDFs.
6. **If a cell fails,** copy the section number and the last 30 lines of the error to Claude Code. Every stage saves to Drive, so after a fix use "Run cell and below" (or Run all: finished stages resume from their saved outputs).
7. **FULL_RESEARCH.** Set `RUN_MODE = "FULL_RESEARCH"`, Run all. About 8-10 h the first time (the SEC download, about 1 h, is cached for later runs). Outputs go to `artifacts_full_research/` (never mixed with QUICK).
8. **Read before you look at test numbers.** Sections 12, 18 and 20 are the validation (selection) results. Section 22 freezes every choice; section 23 opens the test block. Re-running section 23 is logged and disclosed as a re-opening.
9. **Dashboard.** Section 24 checks that the dashboard reproduces the notebook's test predictions, then opens it through Colab's port proxy (works in your browser while the runtime lives). `PUBLIC_TUNNEL = True` gives a temporary public link (anyone with it can use the app).
10. **Reports.** Section 25 writes `reports/NEXUS_Alpha_v3_Computer_Vision_Report.pdf`, `..._NLP_Report.pdf`, `..._Multimodal_Report.pdf` and a results ZIP, and downloads the PDFs.

**Changing a setting changes the stage hash**, so the affected stages recompute automatically. `FORCE_RERUN = True` recomputes everything.

## Phase A addendum (after the full run and the test opening)

1. On Drive, copy the folder `artifacts_full_research` to `artifacts_full_research_v3.0_original` (the confirmatory archive; section 23A.5 compares against it).
2. Upload `NEXUS_Alpha_v3_FULL_PhaseA.ipynb`, GPU runtime (G4 / A100 / L4), High-RAM, Secrets on. Restart session, Run all. Finished stages resume from Drive; expect about 1 h (most of it is section 23A: bootstrap with 5,000 draws and a few LightGBM fits).
3. Outputs: new tables `*_hiB.csv`, `test_H1c/H2c/H3c/H8/H8b/H8c.csv`, `phase_a_*.csv`, `mc_scale_val.csv`; the three PDFs are rebuilt with captions, discussion, conclusion and the post-hoc sections; `results/phase_a_results.json` is the ledger.
4. Check `phase_a_repro_check.csv`: every row must say "identical". If a row says DIFFERENT, stop and send it.
