<!-- ### Price_Predicter

### Software and Tools Requirement

1. [Github Account](https://github.com)
2. [HerokuAccount](https://heroku.com)
3. [ VSCodeIDE ](https://code.visualstudio.com)
4. [GitCLI]()

### creating a new Envirnment
```
python -m venv PRCE_PREDICTER_ENV

```
 -->


# Price Predicter

A machine learning project for predicting house prices based on various features. This project includes a trained model, preprocessing pipeline, and a web interface for predictions.

---

## Software and Tools Required

- [GitHub Account](https://github.com)  
- [Heroku Account](https://www.heroku.com) or [Netlify Account](https://www.netlify.com/)  
- [VS Code IDE](https://code.visualstudio.com)  
- [Git CLI](https://git-scm.com/)  
- Python 3.8+  

---

## Setting Up a New Environment

1. Creating a new Envirnment

```
python -m venv PRCE_PREDICTER_ENV

```

2. Activate the environment:

* **Windows (CMD):**

```bash
PRICE_PREDICTER_ENV\Scripts\activate
```

* **Windows (PowerShell):**

```powershell
.\PRICE_PREDICTER_ENV\Scripts\Activate.ps1
```

* **Linux / macOS:**

```bash
source PRICE_PREDICTER_ENV/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
PRICE_PREDICTOR/
│
├── .git/                        # Git version control directory
├── ENV_PRICE_PREDICTOR/         # Isolated Python virtual environment
│
├── model_training/              # Scripts, data, and resources for model development
│   ├── feature_description.txt  # Documentation describing dataset features
│   ├── HousingData.csv          # The raw or processed dataset
│   └── training_model.py        # Python script for model training
│
├── static/                      # Web-accessible static files
│   └── css/
│       └── styles.css           # Styling for the web interface
│
├── templates/                   # HTML templates for the web interface
│   └── home.html                # Main page for prediction input/output
│
├── .gitignore                   # Specifies files/folders to ignore in Git
├── app.py                       # Main Flask application file
├── LICENSE                      # Project licensing information
├── preprocessing_pipeline.pkl   # Saved scikit-learn preprocessing pipeline
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies for the project
└── trained_model.pkl            # The final serialized machine learning model
```

---

