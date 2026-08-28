# WPSI-OMR-Rank-Prediction
Built by RJ for WPSI candidates


# OMR Evaluation & Analysis Portal

A secure, multi-tier Streamlit application designed for Optical Mark Recognition (OMR) processing, result evaluation, automated Google Sheets logging, and real-time leaderboard analysis.

---

## 🚀 Key Features

* **3-Tier Modular Architecture**: Decoupled public frontend bootstrap loader pulling from a secured private engine repository.
* **Automated OMR Sheet Grading**: Processes uploaded sheets, evaluates Part A and Part B responses, and computes comprehensive scorecards.
* **Google Sheets Integration**: Automatically syncs graded candidate records, leaderboards, and logs via Google Cloud Service Account credentials.
* **Dynamic Candidate Analytics**: Real-time ranking, category-wise breakdowns, and masked roll number exports for secure public sharing.
* **Incentive & Support Dashboard**: Built-in support module with interactive UPI payment integration and supporter showcases.

---

## 🏗️ Architecture Overview

```text
┌─────────────────────────┐
│   Public App Loader     │  (Bootstrap / Entry Point)
│       (app.py)          │
└────────────┬────────────┘
             │ Authenticates via GitHub PAT
             ▼
┌─────────────────────────┐
│   Private Engine Core   │  (Protected Business Logic)
│   (omr-core-engine)     │
└────────────┬────────────┘
             │ Fetches / Updates Records
             ▼
┌─────────────────────────┐
│ Google Cloud Platform   │  (DB / OAuth)
└─────────────────────────┘
