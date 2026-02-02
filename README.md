# 🎬 CineBlend: The Mathematical Movie Matchmaker

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

> **"What happens if you mathematically combine 'Barbie' and 'Oppenheimer'?"**
> CineBlend is a dual-mode recommendation engine that uses Vector Space Modeling to find relationships between movies.

---

## 📺 Project Demo
Click the image below to watch the full walkthrough:

[![CineBlend Demo](https://img.youtube.com/vi/26uhquVzNyg/maxresdefault.jpg)](https://youtu.be/26uhquVzNyg)

---

## 🚀 Features

### 1. Mode A: Classic Recommend
Select a movie (e.g., *Avatar*) and get 5 recommendations based on **Cosine Similarity**.
* **How it works:** Finds vectors with the smallest angular distance to the input movie.
* **Best for:** Finding "more of the same."

### 2. Mode B: The Blend (Signature Feature) 🧬
Select two completely different movies (e.g., *Batman* + *Piranha 3D*).
* **How it works:** Calculates the **Mean Vector** $(\vec{A} + \vec{B}) / 2$ and searches the vector space for movies closest to this new "hypothetical" point.
* **Best for:** Finding hidden gems that bridge the gap between genres.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit + `streamlit-shadcn-ui` (Modern, card-based UI).
* **ML Core:** Scikit-Learn `CountVectorizer` (5000 features).
* **Math:** Cosine Similarity & Vector Arithmetic.
* **Data:** Pandas & NumPy processing on the TMDB 5000 Dataset.

---

## 📦 Installation & Setup

**Note:** This project runs 100% locally. The model files (`.pkl`) are excluded from the repo to save space, so you **must** generate them first.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Dhruv-Mann/CineBlend-ML.git](https://github.com/Dhruv-Mann/CineBlend-ML.git)
    cd CineBlend-ML
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **⚡ Generate the Engine (One-Time Setup)**
    Run the preprocessing script to build the vector models:
    ```bash
    python data_preprocessing.py
    ```
    *Wait until you see "Engine built and saved!"*

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

```text
CineBlend-ML/
├── app.py                  # Main Streamlit Application (Frontend)
├── data_preprocessing.py   # The Engine (builds the .pkl models)
├── requirements.txt        # Dependencies
├── .gitignore             # Ignores large .pkl and .csv files
└── README.md              # Documentation
