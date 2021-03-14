import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
import datetime
from numpy import unique

##Address Here
train_data_path = "C:\\Users\\kirut\\Documents\\Data Storm\\New folder\\data\\Hotel-A-train.csv"

##CSV format here
train_data = pd.read_csv(train_data_path)

y_encoder = OrdinalEncoder(categories=[["Check-In","Canceled","No-Show"]])
y = y_encoder.fit_transform(train_data["Reservation_Status"].to_frame())
print(unique(y,return_counts=True))
input()
num_att = ["Age","Adults","Children","Babies","Room_Rate","Discount_Rate"]
multi_cat = ["Ethnicity","Educational_Level","Country_region","Hotel_Type","Meal_Type","Deposit_type"]
date_att = ["Expected_checkin","Expected_checkout","Booking_date"]
binary_cat = ["Gender","Visted_Previously","Previous_Cancellations","Booking_channel","Required_Car_Parking","Use_Promotion"]
income_att = ["Income"]


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
                values.append(125000)
        values = pd.DataFrame(values)
        return values

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

train_data[income_att] = income_pipeline.fit_transform(train_data[income_att])
train_data[date_att] = date_pipeline.fit_transform(train_data[date_att])
train_data[binary_cat] = OrdinalEncoder().fit_transform(train_data[binary_cat])
train_data["babies_per_adult"] = train_data["Babies"]/train_data["Adults"]
train_data["total_people"] = train_data["Adults"]+train_data["Children"]+train_data["Babies"]
train_data["total_cost"] = train_data["Room_Rate"]*train_data["total_people"]

print()