import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import f1_score,precision_score
import datetime
import pandas as pd
from numpy import transpose,unique,argmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

##Address Here
train_data_path = "C:\\Users\\kirut\\Documents\\Data Storm\\New folder\\data\\Hotel-A-train.csv"
val_data_path = "C:\\Users\\kirut\\Documents\\Data Storm\\New folder\\data\\Hotel-A-validation.csv"
test_data_path = "C:\\Users\\kirut\\Documents\\Data Storm\\New folder\\data\\Hotel-A-test.csv"

##CSV format here
train_data = pd.read_csv(train_data_path)
val_data = pd.read_csv(val_data_path)
test_data = pd.read_csv(test_data_path)

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
num_att = ["Age","Adults","Children","Babies","Room_Rate","Discount_Rate"]
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

train_data_y = y_encoder.fit_transform(train_data_y)

val_data_y = y_encoder.transform(val_data_y)

class_weight = []

_,class_weight = unique(train_data_y,return_counts=True)

class_weight = (sum(class_weight)/class_weight)/3

temp_class_weight = dict()

for i in range(3):
    temp_class_weight[i] = class_weight[i]

classifier = RandomForestClassifier(n_jobs=-1,class_weight=temp_class_weight,max_depth=5)

param_grid = [
   'n_estimators','max_features','max_leaf_nodes','bootstrap'
]

grid_search = RandomizedSearchCV(classifier, param_grid, cv=5,
    scoring=f1_score,
    return_train_score=True)

grid_search.fit(train_preprocessed,train_data_y.ravel())

print(grid_search.best_params_)

classifier = grid_search.best_estimator_
classifier.fit(train_preprocessed,train_data_y.ravel())

print("Score is {}".format(classifier.score(val_preprocessed,val_data_y.ravel())))
test_data_y = classifier.predict(val_preprocessed)


print("Score is {}".format(f1_score(val_data_y,test_data_y,average='macro')))
#test_data_y = classifier.predict(test_preprocessed)


print(unique(test_data_y,return_counts=True))
print(unique(val_data_y,return_counts=True))