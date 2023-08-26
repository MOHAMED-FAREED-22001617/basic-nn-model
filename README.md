# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/MOHAMED-FAREED-22001617/basic-nn-model/assets/121412904/2d05c05b-fc6d-402f-b876-d32447df35cf)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Developed by: MOHAMED FAREED F
Register No : 212222230082
```
### Importing Modules
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default
```
### Authenticate & Create Dataframe using Data in Sheets
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('MyMLData').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})

dataset1.head()
```
### Assign X and Y values
```
X = dataset1[['Input']].values
y = dataset1[['Output']].values

X
y
```
### Normalize the values & Split the data
```
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
X_train1
```
## Create a Neural Network & Train it:

Create the model
```
ai=Sequential([
    Dense(7,activation='relu'),
    Dense(14,activation='relu'),
    Dense(1)
])
```
Compile the model
```
ai.compile(optimizer='rmsprop',loss='mse')
```
Fit the model
```
ai.fit(X_train1,y_train,epochs=2000)
ai.fit(X_train1,y_train,epochs=2000)
```
### Plot the Loss
```
loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()
```
### Evaluate the model
```
X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)
```
### Predict for some value
```
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)
```


## Dataset Information

![image](https://github.com/MOHAMED-FAREED-22001617/basic-nn-model/assets/121412904/85c53eb8-70b6-4f6d-bdce-480703e345f9)

## OUTPUT


### Training Loss Vs Iteration Plot

![image](https://github.com/MOHAMED-FAREED-22001617/basic-nn-model/assets/121412904/e9e5e26f-2ff0-4faa-85dd-248c85485971)

### Test Data Root Mean Squared Error

![image](https://github.com/MOHAMED-FAREED-22001617/basic-nn-model/assets/121412904/9eef8a55-f0ca-40fc-af9a-3d065e4b89df)


### New Sample Data Prediction
![image](https://github.com/MOHAMED-FAREED-22001617/basic-nn-model/assets/121412904/5f866661-3b11-4c80-a8e9-0e9085f2db76)


## RESULT
Thus the neural network regression model for the given dataset is developed and executed successfully.
