🔧 Installation & Setup
1️⃣ Clone the Repository
git clone <your-repo-url>
cd Customer_Churn_Modelling

2️⃣ Create a Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Mac/Linux
# OR
.venv\Scripts\activate  # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Data Pipeline & Train the Model
python src/Components/Data_ingestion.py
python src/Components/Data_transformation.py
python src/Components/Model_trainer.py

5️⃣ Run Flask App
python app.py
🔹 Open http://127.0.0.1:5000/ in your browser.

💻 Using the Web App
1️⃣ Go to the homepage (index.html)
2️⃣ Select input features from dropdowns
3️⃣ Click "Predict"
4️⃣ See the churn prediction on result.html

📊 Model Details
Algorithm: XGBoost Classifier
Accuracy: ~81%
Feature Transformation: Ordinal Encoding + OneHot Encoding
Class Imbalance Handling: scale_pos_weight

📌 Future Enhancements
✅ Improve UI with Bootstrap
✅ Deploy on AWS Lambda or Render
✅ Optimize Model Performance

📜 License
This project is open-source and free to use.


