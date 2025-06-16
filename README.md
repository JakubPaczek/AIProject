# Ocena ryzyka kredytowego

Prognozowanie zdolności kredytowej klientów banku, z wykorzystaniem danych z Kaggle i klasycznych modeli ML.

---

## 1. Opis zbioru danych

- Zbiór zawiera **150 000 wierszy** i **10 cech numerycznych** + etykieta (`SeriousDlqin2yrs`).
- Przegląd danych i przykład kilku wierszy.
- Identyfikacja braków danych (`MonthlyIncome`, `NumberOfDependents`).
- Statystyki opisowe pozwalają wykryć outliery:
  - `age=0`
  - wartości `98` i `96` w zmiennych liczby opóźnień
  - procentowe cechy (`DebtRatio`, `RevolvingUtilizationOfUnsecuredLines`) przekraczające `1.0`
- Wizualizacje: histogramy, macierz korelacji.

---

##  2. Preprocessing

- Usunięcie nieprawidłowych wartości i imputacja braków danych (`median()`).
- Korekta wartości odstających (np. `age < 18`, `% > 1.0`).
- Podział danych na zbiór treningowy/testowy z **`train_test_split(..., stratify=y)`**.
- Skalowanie danych z użyciem **`StandardScaler`**.

---

## 3. Trening modeli i walidacja krzyżowa

- Trening wielu modeli z użyciem **`Pipeline` + `cross_val_score` (5-fold CV)**.
- Metryki: `F1 Score`, `ROC AUC`.
- Wybrane modele (o najlepszych wynikach):  
  - `LogisticRegression`  
  - `XGBClassifier`
- **GridSearchCV** dla strojenia hiperparametrów.
- Powtórny trening najlepszych modeli.
- Wizualizacje:
  - `Confusion Matrix`
  - `ROC Curve`
  - `Precision-Recall Curve`

---

## 4. Podsumowanie i interpretacja

- **Feature Importance**:
  - `coef_` (dla `LogisticRegression`)
  - `feature_importances_` (dla `XGBoost`)
- Wnioski:
  - Najważniejsze czynniki wpływające na ryzyko kredytowe
  - Porównanie modeli i skuteczność predykcji
