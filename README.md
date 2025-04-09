# 📊 Postech Tech Challenge 1 – Health Cost Prediction

This project explores a dataset containing medical and demographic data to understand patterns related to **health charges** and predict them using **Machine Learning models**.

---

## 🧠 Objective

Predict **medical charges** based on key attributes such as:

- Age
- Gender
- BMI (Body Mass Index)
- Number of children
- Smoking status
- Region

---

## 🔎 Exploratory Data Analysis (EDA)

We began by exploring the dataset's distribution:

- ✔️ No missing or duplicated data
- ✔️ Balanced gender distribution
- ✔️ Higher average charges for **smokers** than **non-smokers**
- ✔️ Slightly higher charges for **males**
- ✔️ BMI tends to correlate with higher charges
- ✔️ Most participants have 1–2 children

We used visualizations (bar charts, scatter plots) to show distributions and relationships between variables like gender, smoking status, region, and BMI vs. charges.

---

## 🧼 Preprocessing

We applied **Label Encoding** to convert categorical features:

```python
from sklearn.preprocessing import LabelEncoder
df['gênero'] = label_encoder.fit_transform(df['gênero'])
df['fumante'] = label_encoder.fit_transform(df['fumante'])
df['região'] = label_encoder.fit_transform(df['região'])
```

## 🔗 Correlation Matrix

We computed the Pearson correlation between all features in the dataset to identify the strongest relationships with the target variable `encargos` (medical charges).

| Feature   | Correlation with Encargos |
|-----------|----------------------------|
| idade     | **0.51**                   |
| imc       | **0.49**                   |
| filhos    | **0.32**                   |
| fumante   | **0.42**                   |
| gênero    | 0.22                       |
| região    | 0.19                       |

🔍 **Insight:** The variables that most influence `encargos` are `idade`, `imc`, `filhos`, and `fumante`.

---

## 🔀 Data Splitting

We selected the top 4 correlated features to use as inputs (`X`) and defined `encargos` as the target (`y`). Then we split the dataset into training and test sets.

```python
x = df[['fumante', 'idade', 'imc', 'filhos']]
y = df['encargos']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

## 🤖 Models Tested

We evaluated three regression models to predict medical charges based on the most correlated features: **smoker status**, **age**, **BMI**, and **number of children**.

---

### 1. 📈 Linear Regression
- ✅ **Best overall performance**
- **R² (cross-validation):** 0.6019
- **R² (test set):** 0.7560
- **Mean Squared Error (MSE):** 3.68M
- **Mean Absolute Error (MAE):** 1.46K
- Performs well due to the linear relationship among features

---

### 2. 🌳 Decision Tree Regressor
- **R² (cross-validation):** 0.1683
- Higher variance and overfitting tendencies
- Performance fluctuated across folds
- Suitable for non-linear data, but not ideal here

---

### 3. 🌲 Random Forest Regressor
- **R² (cross-validation):** 0.5244
- Better generalization than Decision Tree
- More robust but still underperformed compared to Linear Regression
- Potential for improvement with tuning

---

🔍 **Result:**  
Among the three, **Linear Regression** achieved the best balance between performance and consistency, making it the most appropriate model for this dataset.

## 📊 Cross-Validation Comparison

To ensure the reliability of our results, we applied **K-Fold Cross-Validation** with 5 folds and compared the performance (R² score) of all three models:

| Model              | Mean R² Score |
|-------------------|---------------|
| Linear Regression | **0.6019**    |
| Decision Tree     | 0.1683        |
| Random Forest     | 0.5244        |

- The cross-validation results confirmed that **Linear Regression** offers the best generalization for this dataset.
- Although **Random Forest** performed slightly better than Decision Tree, it still fell short of Linear Regression in terms of overall R² score.

🏆 **Conclusion:**  
**Linear Regression** is the most suitable model for this challenge based on its consistent cross-validation performance.

## 📈 Final Prediction (Test Set)

After determining that **Linear Regression** was the best-performing model, we used it to make predictions on the test set.

### 🔢 Evaluation Metrics:

- **Mean Squared Error (MSE):** `3,681,978.83`
- **Mean Absolute Error (MAE):** `1,458.07`
- **R² Score:** **0.756**

These metrics indicate that the model performs well in estimating the medical charges, explaining around **75.6%** of the variance in the test data.

---

### 📊 Real vs Predicted Values

We visualized the model’s performance by plotting actual values against predicted values:

```python
plt.scatter(y_test, previsoes, label='Real vs. Predicted', color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Real vs Predicted Medical Charges')
plt.legend()
plt.show()
```

