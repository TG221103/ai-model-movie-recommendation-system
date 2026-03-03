# 🎬 Movie Recommendation System

A machine learning-based full-stack web application that takes a user's favorite movie as input and recommends 5 similar movies based on genres, keywords, tagline, cast, and director.

## Features
- Content-Based Filtering using TF-IDF and Cosine Similarity.
- Interactive Web UI built with Streamlit.
- Device-independent execution out of the box.
- Error handling for invalid inputs.
- Only requires Python and basic ML dependencies. CPU-only.

---

## 🚀 Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8 or newer installed on your system.
- You can check your Python version by running:
  ```bash
  python --version
  ```

### 2. Set Up a Virtual Environment 
It's highly recommended to use a virtual environment so the project dependencies don't conflict with your global Python packages.

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Run the following command to securely install all the required Python libraries using the project's requirements file:

```bash
pip install -r requirements.txt
```

---

## 🏃‍♀️ How to Run

### Step 1: Train the Model (First Time Only)
We must process the dataset `movies.csv` and generate the serialized model artifacts (`model.pkl`):
```bash
python train_model.py
```
*You should see a message saying "Model training and saving completed successfully!"*

### Step 2: Start the Web App
Run the user-friendly interface using Streamlit:
```bash
python -m streamlit run app.py
```
This will start a local web server, and a browser window should automatically open with the app. You can also manually access it via `http://localhost:8501`.

---

## 🛠️ Project Structure
- `train_model.py`: Script to process `movies.csv`, calculate TF-IDF vectors, and save the data logic to `model.pkl`.
- `app.py`: Streamlit application file that houses the UI and dynamically returns identical movie recommendations.
- `movies.csv`: Original movie dataset (make sure this is always in the same directory as the scripts).
- `requirements.txt`: Master list of dependencies mapped out smoothly for compatibility.

## ⚠️ Troubleshooting & Common Issues
- **`ModuleNotFoundError`**: This indicates that the necessary packages were not installed perfectly or the virtual environment is not activated. Ensure you ran `pip install -r requirements.txt` while `(venv)` is active in your terminal.
- **Dataset not found error**: Ensure `movies.csv` is correctly placed in the same main folder as `train_model.py`.
- **Model file not found error in Streamlit**: You must run `python train_model.py` before `streamlit run app.py` can work properly.


