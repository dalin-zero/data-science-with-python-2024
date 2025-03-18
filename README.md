
# **การใช้ CRISP-DM ในการพยากรณ์ราคาหุ้น AOT ปี 2024**

## **1. Business Understanding**  
CRISP-DM (Cross-Industry Standard Process for Data Mining) เป็นกระบวนการมาตรฐานที่ใช้ในงาน Data Science  
ในโครงการนี้ เราต้องการพยากรณ์ราคาหุ้นของ **AOT** ในปี **2024** โดยใช้ **LSTM (Long Short-Term Memory)** ซึ่งเป็นโมเดล Deep Learning สำหรับ Time Series

**วัตถุประสงค์**  
- วิเคราะห์แนวโน้มราคาหุ้น AOT  
- ใช้ข้อมูลในอดีตเพื่อสร้างแบบจำลองทำนายอนาคต  
- นำเสนอผลลัพธ์ที่สามารถใช้เป็นแนวทางการลงทุน  

---

## **2. Data Understanding**  
เราจะใช้ข้อมูลราคาหุ้น AOT ที่ดึงมาจาก **Yahoo Finance** ผ่าน `yfinance` โดยมีคอลัมน์ที่สำคัญดังนี้:  

- `Date` - วันที่ซื้อขาย  
- `Open` - ราคาเปิด  
- `High` - ราคาสูงสุด  
- `Low` - ราคาต่ำสุด  
- `Close` - ราคาปิด  
- `Volume` - ปริมาณการซื้อขาย  

ตัวอย่างข้อมูล:  
```python
import yfinance as yf

# ดึงข้อมูลหุ้น AOT
df = yf.download("AOT.BK", start="2010-01-01", end="2023-12-31")
df.head()
```

---

## **3. Data Preparation**  
### **3.1 การทำความสะอาดข้อมูล**  
```python
# ตรวจสอบข้อมูลที่หายไป
print(df.isnull().sum())
```
หากพบค่าว่าง เราจะทำการลบหรือเติมค่าตามความเหมาะสม

### **3.2 การปรับขนาดข้อมูล**  
```python
from sklearn.preprocessing import MinMaxScaler

# ใช้เฉพาะราคาปิด
data = df[["Close"]].values

# สเกลให้อยู่ในช่วง 0-1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
```

### **3.3 การสร้างชุดข้อมูลแบบ Time Series**  
```python
import numpy as np

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# ตั้งค่าช่วงเวลา 60 วันย้อนหลัง
time_step = 60
X, Y = create_dataset(scaled_data, time_step)
```

### **3.4 แบ่งข้อมูล Train-Test**  
```python
# แบ่ง Train-Test (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# แปลงข้อมูลให้อยู่ในรูปแบบที่ LSTM ต้องการ
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
```

---

## **4. Modeling and Training**  
เราจะใช้ **LSTM** เพื่อสร้างแบบจำลองทำนายราคาหุ้น  

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# สร้างโมเดล LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# คอมไพล์โมเดล
model.compile(optimizer="adam", loss="mean_squared_error")

# เทรนโมเดล
model.fit(X_train, Y_train, batch_size=16, epochs=50, verbose=1)
```

---

## **5. การทำนายหุ้น AOT ปี 2024**  
เราจะใช้โมเดลที่เทรนแล้วมาพยากรณ์ราคาหุ้น AOT ในปี 2024  

```python
future_days = 365  # จำนวนวันที่ต้องการทำนาย
future_predictions = []

# ใช้ข้อมูล 60 วันสุดท้ายเป็นจุดเริ่มต้น
last_60_days = scaled_data[-time_step:].reshape(1, time_step, 1)

for _ in range(future_days):
    pred = model.predict(last_60_days, verbose=0)
    pred = pred.reshape(1, 1, 1)  # แปลงให้มีขนาด 3 มิติ
    future_predictions.append(pred[0, 0])
    last_60_days = np.append(last_60_days[:, 1:, :], pred, axis=1)

# แปลงค่ากลับเป็นราคาหุ้นจริง
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# สร้างช่วงวันที่สำหรับปี 2024
last_date = pd.to_datetime(df.index[-1])
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
```

---

## **6. Visualization of Prediction**  
เราสามารถแสดงกราฟพยากรณ์ราคาหุ้น AOT ปี 2024  

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(future_dates, future_predictions, label="Predicted AOT Stock Price")
plt.xlabel("Date")
plt.ylabel("Stock Price (THB)")
plt.title("Predicted AOT Stock Price for 2024")
plt.legend()
plt.show()
```

---

## **7. สรุปผลการวิเคราะห์**  
- โมเดล **LSTM** สามารถพยากรณ์แนวโน้มราคาหุ้น AOT ในปี 2024 ได้  
- การใช้ข้อมูลย้อนหลัง 60 วันช่วยให้โมเดลเรียนรู้แนวโน้มได้ดีขึ้น  
- สามารถนำโมเดลนี้ไปใช้กับหุ้นตัวอื่นได้โดยเปลี่ยนข้อมูลอินพุต  

**หมายเหตุ:** การทำนายราคาหุ้นมีความไม่แน่นอนสูง ควรใช้ร่วมกับปัจจัยทางเศรษฐกิจและปัจจัยพื้นฐานของบริษัท  
