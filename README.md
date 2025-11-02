# Logistics Delivery Delay - EDA and Prediction
### Predicting Delivery Delays and Enhancing Supply Chain Efficiency

---

## **Introduction**
This project focuses on predicting delivery delays in logistics using the publicly available [Logistics Data Containing Real-World Data](https://www.kaggle.com/datasets/pushpitkamboj/logistics-data-containing-real-world-data) dataset.  
By analyzing shipment patterns and delivery characteristics, this project aims to help supply chain professionals anticipate potential delays, optimize operations, and improve customer satisfaction through data-driven insights.  

---

## **Live App**
Experience the interactive **Delivery Delay Prediction Dashboard** here:  
👉 [**Launch Streamlit App**]((https://logistics-delivery-delay---eda-and-prediction-mu8xttxw7udcwdpp.streamlit.app/))

---


**Key highlights of the project:**  
- Comprehensive **Exploratory Data Analysis (EDA)** to identify trends and delay drivers.  
- **Feature engineering** for shipment, product, and time-related attributes.  
- **SMOTE oversampling** applied to handle class imbalance across delivery outcomes.  
- **Stacking Classifier** combining **Random Forest** and **XGBoost** as base models with **Logistic Regression** as the meta-model.  
- Performance evaluation using **accuracy**, **precision**, **recall**, and **F1-score** metrics.  

**SEO Keywords:** logistics delay prediction, supply chain analytics, machine learning for logistics, EDA, stacking classifier, Random Forest, XGBoost, SMOTE, predictive modeling, delivery performance.

---

## **Visual Representations**

### **1. Delivery Outcome Distribution**
![Delivery Outcome Distribution](Images/Delivery%20Outcome%20Distribution.png)

### **2. Order Volume per Month by Delivery Outcome**
![Order Volume per Month by Delivery Outcome](Images/Order%20Volume%20per%20Month%20by%20Delivery%20Outcome.png)

### **3. Delivery Outcome by Shipping Mode**
![Delivery Outcome by Shipping Mode](Images/Delivery%20Outcome%20by%20Shipping%20Mode.png)

### **4. Distribution of Delivery Time (Days) by Region**
![Distribution of Delivery Time (Days) by Region](Images/Distribution%20of%20Delivery%20Time%20(Days)%20by%20Region.png)

---

## **Project Structure**
```
Logistics Delivery Delay - EDA and Prediction/
│
├── Data/
│   └── logistics_data.csv
│
├── Images/
│   ├── Delivery Outcome Distribution.png
│   ├── Order Volume per Month by Delivery Outcome.png
│   ├── Delivery Outcome by Shipping Mode.png
│   └── Distribution of Delivery Time (Days) by Region.png
│
├── Notebooks/
│   └── Delivery Delay Prediction.ipynb
│
├── requirements.txt
└── README.md
```

---

## **Installation & Usage**

1. **Clone the repository**
```bash
git clone https://github.com/rAiswariya/Logistics-Delivery-Delay-EDA-and-Prediction.git
cd Logistics-Delivery-Delay-EDA-and-Prediction
```

2. **Set up the environment**
```bash
pip install -r requirements.txt
```

3. **Open the Notebook**
* Open `Delivery Delay Prediction.ipynb` in [Google Colab](https://colab.research.google.com/) or locally in Jupyter Notebook.  
* Ensure the `Data` and `Images` folders are in the same directory as the notebook.

4. **Run the Notebook**
* Execute all cells to perform EDA, feature engineering, model training, and evaluation.

---

## **Known Issues**

* **Feature Enrichment:** External factors such as traffic, weather, or holiday data are not yet included but could enhance predictions.  
* **Real-time Prediction:** A Streamlit-based dashboard for live delivery delay forecasting will be added in future updates.

---

## **Contributing**

Contributions are welcome!

* If you find bugs, issues, or improvements, please submit them via the **Issues** tab.  
* Pull requests are appreciated for adding features, improving visualizations, or enhancing modeling techniques.  
* Ensure that added code is properly documented and reproducible.  

> Let’s collaborate to make this delivery delay prediction and supply chain optimization project even better!

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
