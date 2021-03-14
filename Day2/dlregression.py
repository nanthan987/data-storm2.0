import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import f1_score
import datetime
import pandas as pd
from numpy import transpose,argmax,unique,sum,log,average,array,int32
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Input,GaussianDropout,BatchNormalization,Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.regularizers import l1_l2,l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam
from  tensorflow.keras.metrics import Accuracy
from tensorflow.keras.initializers import Constant

##Address Here
train_data_path = "C:\\Users\\kirut\\Documents\\Data Storm\\New folder\\data\\Hotel-A-train.csv"
val_data_path = "C:\\Users\\kirut\\Documents\\Data Storm\\New folder\\data\\Hotel-A-validation.csv"
test_data_path = "C:\\Users\\kirut\\Documents\\Data Storm\\New folder\\data\\Hotel-A-test.csv"

##CSV format here
train_data = pd.read_csv(train_data_path)
val_data = pd.read_csv(val_data_path)
test_data = pd.read_csv(test_data_path)

##Add new columns
train_data["babies_per_adult"] = train_data["Babies"]/train_data["Adults"]
val_data["babies_per_adult"] = val_data["Babies"]/val_data["Adults"]
test_data["babies_per_adult"] = test_data["Babies"]/test_data["Adults"]

train_data["total_people"] = train_data["Adults"]+train_data["Children"]+train_data["Babies"]
train_data["total_cost"] = train_data["Room_Rate"]*train_data["total_people"]
val_data["total_people"] = val_data["Adults"]+val_data["Children"]+train_data["Babies"]
val_data["total_cost"] = val_data["Room_Rate"]*val_data["total_people"]
test_data["total_people"] = test_data["Adults"]+test_data["Children"]+train_data["Babies"]
test_data["total_cost"] = test_data["Room_Rate"]*test_data["total_people"]

##Extract Y feature here
train_data_y = train_data["Reservation_Status"].to_frame()
train_data.drop("Reservation_Status",axis="columns",inplace=True)

val_data_y = val_data["Reservation_Status"].to_frame()
val_data.drop("Reservation_Status",axis="columns",inplace=True)

##Remove Reservation ID
train_data.drop("Reservation-id",axis="columns",inplace=True)
val_data.drop("Reservation-id",axis="columns",inplace=True)
test_data.drop("Reservation-id",axis="columns",inplace=True)

##Categorising Attributes based on preprocessing properties
num_att = ["Age","Adults","Children","Babies","Room_Rate","Discount_Rate","babies_per_adult","total_cost","total_people"]
multi_cat = ["Ethnicity","Educational_Level","Country_region","Hotel_Type","Meal_Type","Deposit_type"]
date_att = ["Expected_checkin","Expected_checkout","Booking_date"]
binary_cat = ["Gender","Visted_Previously","Previous_Cancellations","Booking_channel","Required_Car_Parking","Use_Promotion"]
income_att = ["Income"]

##Defining Preprocessing Classes

class date_preprocessing(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.margin = datetime.datetime.now()
    def fit(self,X,y=None):
        X = X.values.tolist() ##Shape (rows,3)
        self.margin = []
        for x in X[0]:
            self.margin.append(datetime.datetime.strptime(x,'%m/%d/%Y'))
        return self
    def transform(self,X,y=None):
        X = X.values.tolist()
        delta = []
        for x in X:
            temp_delta = []
            for date in range(len(x)):
                temp_delta.append((datetime.datetime.strptime(x[date],'%m/%d/%Y')-self.margin[date]).days)
            delta.append(temp_delta)
        delta = pd.DataFrame(delta)
        return delta

class income_preprocessing(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.l1 = "<25K"
        self.l2 = "25K --50K"
        self.l3 = "50K -- 100K"
        self.l4 = ">100K"

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X = X.values.tolist()
        values = []
        for x in X:
            if x == self.l1:
                values.append(12500)
            elif x == self.l2:
                values.append(37500)
            elif x == self.l3:
                values.append(75000)
            else:
                values.append(100000)
        values = pd.DataFrame(values)
        return values

##Defining Pipelines Here
num_pipeline = Pipeline([
    ('min_max',MinMaxScaler())
])

date_pipeline = Pipeline([
    ("int_conv",date_preprocessing()),
    ('std_scaler', StandardScaler()),
])

income_pipeline = Pipeline([
    ("int_conv",income_preprocessing()),
    ('min_max',MinMaxScaler())
])

full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_att),
    ("cat",OneHotEncoder(),multi_cat),
    ("binary",OrdinalEncoder(),binary_cat),
    ("date",date_pipeline,date_att),
    ("income",income_pipeline,income_att)
])

##Fit And transform Train Data
train_preprocessed = full_pipeline.fit_transform(train_data)

##Fit and Transform Validation Data
val_preprocessed = full_pipeline.transform(val_data)

test_preprocessed = full_pipeline.transform(test_data)

##Transform Y features
y_encoder = OrdinalEncoder(categories=[["Check-In","Canceled","No-Show"]])

train_data_y = y_encoder.fit_transform(train_data_y)/3

val_data_y = y_encoder.transform(val_data_y)/3

class_weight = []

bias_weight = average(train_data_y)

classifier = Sequential([
    Input(train_preprocessed.shape[1]),
    GaussianDropout(0.25),
    Dense(256,activation="softplus",kernel_regularizer=l1_l2(l2=0.001)),
    BatchNormalization(),
    Dropout(0.25),
    Dense(128,activation="softplus",kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.25),
    Dense(64,activation="softplus",kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.25),
    Dense(32,activation="softplus",kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.125),
    Dense(16,activation="softplus",kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.0625),
    Dense(8,activation="softplus",kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dense(1,activation="sigmoid",bias_initializer=Constant(bias_weight))
])

print(classifier.summary())

classifier.compile(optimizer=Nadam(1e-2),loss="MAE")

history = classifier.fit(
    train_preprocessed,train_data_y,
    epochs=100,batch_size=64,
    validation_data=(val_preprocessed,val_data_y),
    workers=-1,use_multiprocessing=True,
    callbacks=[EarlyStopping(patience=20,restore_best_weights=True),ReduceLROnPlateau()])

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()

classifier.evaluate(val_preprocessed,val_data_y,batch_size=None)

test_data_y = array(classifier.predict(train_preprocessed),int32)

print(unique(train_data_y,return_counts=True))
print(unique(test_data_y,return_counts=True))
acc = Accuracy()
acc.update_state(test_data_y,train_data_y)
print(acc.result().numpy())