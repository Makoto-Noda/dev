#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse


# In[ ]:


def data_pre(df):
    
    #結果に関与しなさそうな行ドロップ
    df = df.drop("description", axis=1)
    df = df.drop("name", axis=1)
    df = df.drop("thumbnail_url", axis=1)

    #"neighbourhood""property_type"もドロップ
    df = df.drop("property_type", axis=1)

    #特徴量重要度が低い”bed_type””host_has_profile_pic”"instant_bookable"もドロップ
    df = df.drop("bed_type", axis=1)
    df = df.drop("host_has_profile_pic", axis=1)
    df = df.drop("instant_bookable", axis=1)

    #"first_review"は"host_since"と相関強いためドロップ
    df = df.drop("first_review", axis=1)

    #"latitude""longitude""city"ドロップ
    df = df.drop("latitude", axis=1)
    df = df.drop("longitude", axis=1)
    df = df.drop("city", axis=1)
    
    df["neighbourhood"] = df["neighbourhood"].astype("category")
    
    #"cleaning_fee""host_identity_verified""cancellation_policy""room_type"をカテゴリ化
    bool_dic = {"t":1, "f":0}
    df["cleaning_fee"] = df["cleaning_fee"].replace(bool_dic).astype("category")

    bool_dic = {"t":1, "f":0}
    df["host_identity_verified"] = df["host_identity_verified"].replace(bool_dic).astype("category")

    cancel_dic = {'flexible':0, 'moderate':1, 'strict':2, 'super_strict_30':3, 'super_strict_60':4}
    df["cancellation_policy"] = df["cancellation_policy"].replace(cancel_dic).astype("category")

    room_dic = {'Entire home/apt':0, 'Private room':1, 'Shared room':2}
    df["room_type"] = df["room_type"].replace(room_dic).astype("category")
      
    #"host_since""last_review"カラムを数値に変換。欠損値はpandasのisnullで判定してmeanを代入。
    host_since_date_dic = {}
    for i in df["host_since"]:
        if pd.isnull(i):
            date_int = 2.014043e+07
            host_since_date_dic[i] = date_int
        else:
            date = str(i)
            date_int = int(date.replace("-", ""))
            host_since_date_dic[i] = date_int
    df["host_since"] = df["host_since"].replace(host_since_date_dic).astype(int)

    last_review_date_dic = {}
    for i in df["last_review"]:
        if pd.isnull(i):
            date_int = 2.016912e+07
            last_review_date_dic[i] = date_int
        else:
            date = str(i)
            date_int = int(date.replace("-", ""))
            last_review_date_dic[i] = date_int
    df["last_review"] = df["last_review"].replace(last_review_date_dic).astype(int)
    
    #"host_response_rate"も末尾の%取ってint型に
    host_response_rate_dic = {}
    for i in df["host_response_rate"]:
        if pd.isnull(i):
            rate_int = 95.598956
            host_response_rate_dic[i] = rate_int
        else:
            rate = i.rstrip("%")
            rate_int = int(rate)
            host_response_rate_dic[i] = rate_int
    df["host_response_rate"] = df["host_response_rate"].replace(host_response_rate_dic).astype(int)
    
    #review_scores_ratingもNaNをmean埋め
    review_scores_rating_dic = {}
    df["review_scores_rating"] = df["review_scores_rating"].astype(float)
    for i in df["review_scores_rating"]:
        if pd.isnull(i):
            review_scores_rating_float = 94.077928
            review_scores_rating_dic[i] = review_scores_rating_float
        else:
            review_scores_rating_float = float(i)
            review_scores_rating_dic[i] = review_scores_rating_float
    df["review_scores_rating"] = df["review_scores_rating"].replace(review_scores_rating_dic)
    
    #"zipcode"をint型に
    zipcode_dic = {}
    for i in df["zipcode"]:
        if pd.isnull(i):
            zipcode = 0
            zipcode_dic[i] = zipcode
        elif " " in i:
            zipcode = 0
            zipcode_dic[i] = zipcode
        elif "1m" in i:
            zipcode = 0
            zipcode_dic[i] = zipcode
        elif "7302." in i:
            zipcode = 0
            zipcode_dic[i] = zipcode
        else:
            zipcode = i[0:5]
            zipcode_dic[i] = zipcode
    df["zipcode"] = df["zipcode"].replace(zipcode_dic).astype(int)
    df["zipcode"] = df["zipcode"].astype("category")
    
    #"amenities"品目数をスコア化。replaceに時間かかるので最後にやる。
    rep_dic = {}
    for i in df["amenities"]:
        rep = i.replace("{", "")
        rep = rep.replace("}", "")
        rep = rep.replace(" ", "")
        rep_list = sorted(rep.split(","))
        score = int(len(rep_list)) / 2
        rep_dic[i] = score
    df["amenities"] = df["amenities"].replace(rep_dic)
    
    return df   


# In[ ]:


def data_pre_test(df):
    
    #結果に関与しなさそうな行ドロップ
    df = df.drop("description", axis=1)
    df = df.drop("name", axis=1)
    df = df.drop("thumbnail_url", axis=1)

    #"neighbourhood""property_type"もドロップ
    df = df.drop("property_type", axis=1)

    #特徴量重要度が低い”bed_type””host_has_profile_pic”"instant_bookable"もドロップ
    df = df.drop("bed_type", axis=1)
    df = df.drop("host_has_profile_pic", axis=1)
    df = df.drop("instant_bookable", axis=1)

    #"first_review"は"host_since"と相関強いためドロップ
    df = df.drop("first_review", axis=1)

    #"latitude""longitude""city"ドロップ
    df = df.drop("latitude", axis=1)
    df = df.drop("longitude", axis=1)
    df = df.drop("city", axis=1)

    df["neighbourhood"] = df["neighbourhood"].astype("category")
    
    #"cleaning_fee""host_identity_verified""cancellation_policy""room_type"をカテゴリ化
    bool_dic = {"t":1, "f":0}
    df["cleaning_fee"] = df["cleaning_fee"].replace(bool_dic).astype("category")

    bool_dic = {"t":1, "f":0}
    df["host_identity_verified"] = df["host_identity_verified"].replace(bool_dic).astype("category")

    cancel_dic = {'flexible':0, 'moderate':1, 'strict':2, 'super_strict_30':3, 'super_strict_60':4}
    df["cancellation_policy"] = df["cancellation_policy"].replace(cancel_dic).astype("category")

    room_dic = {'Entire home/apt':0, 'Private room':1, 'Shared room':2}
    df["room_type"] = df["room_type"].replace(room_dic).astype("category")
    
    #"host_since""last_review"カラムを数値に変換。欠損値はpandasのisnullで判定してmeanを代入。
    host_since_date_dic = {}
    for i in df["host_since"]:
        if pd.isnull(i):
            date_int = 2.014043e+07
            host_since_date_dic[i] = date_int
        else:
            date = str(i)
            date_int = int(date.replace("-", ""))
            host_since_date_dic[i] = date_int
    df["host_since"] = df["host_since"].replace(host_since_date_dic).astype(int)

    last_review_date_dic = {}
    for i in df["last_review"]:
        if pd.isnull(i):
            date_int = 2.016912e+07
            last_review_date_dic[i] = date_int
        else:
            date = str(i)
            date_int = int(date.replace("-", ""))
            last_review_date_dic[i] = date_int
    df["last_review"] = df["last_review"].replace(last_review_date_dic).astype(int)
    
    #"host_response_rate"も末尾の%取ってint型に
    host_response_rate_dic = {}
    for i in df["host_response_rate"]:
        if pd.isnull(i):
            rate_int = 95.598956
            host_response_rate_dic[i] = rate_int
        else:
            rate = i.rstrip("%")
            rate_int = int(rate)
            host_response_rate_dic[i] = rate_int
    df["host_response_rate"] = df["host_response_rate"].replace(host_response_rate_dic).astype(int)
    
    #review_scores_ratingもNaNをmean埋め
    review_scores_rating_dic = {}
    df["review_scores_rating"] = df["review_scores_rating"].astype(float)
    for i in df["review_scores_rating"]:
        if pd.isnull(i):
            review_scores_rating_float = 94.077928
            review_scores_rating_dic[i] = review_scores_rating_float
        else:
            review_scores_rating_float = float(i)
            review_scores_rating_dic[i] = review_scores_rating_float
    df["review_scores_rating"] = df["review_scores_rating"].replace(review_scores_rating_dic)
    
    #"zipcode"をint型に
    zipcode_dic = {}
    for i in df["zipcode"]:
        if pd.isnull(i):
            zipcode = 0
            zipcode_dic[i] = zipcode
        elif " " in i:
            zipcode = 0
            zipcode_dic[i] = zipcode
        elif "1m" in i:
            zipcode = 0
            zipcode_dic[i] = zipcode
        elif "7302." in i:
            zipcode = 0
            zipcode_dic[i] = zipcode
        else:
            zipcode = i[0:5]
            zipcode_dic[i] = zipcode
    df["zipcode"] = df["zipcode"].replace(zipcode_dic).astype(int)
    df["zipcode"] = df["zipcode"].astype("category")
    
    #"amenities"品目数をスコア化。replaceに時間かかるので最後にやる。
    rep_dic = {}
    for i in df["amenities"]:
        rep = i.replace("{", "")
        rep = rep.replace("}", "")
        rep = rep.replace(" ", "")
        rep_list = sorted(rep.split(","))
        score = int(len(rep_list)) / 2
        rep_dic[i] = score
    df["amenities"] = df["amenities"].replace(rep_dic)
    
    return df   


# In[ ]:


def model_lgb(df):

    df_train, df_val = train_test_split(df, test_size=0.2)
    
    col = "y"
    train_y = df_train[col]
    train_x = df_train.drop(col, axis=1)

    val_y = df_val[col]
    val_x = df_val.drop(col, axis=1)

    trains = lgb.Dataset(train_x, train_y)
    valids = lgb.Dataset(val_x, val_y)
    
    #airbnb_param_optimizationで得た値をハイパーパラメータに適用
    #https://lightgbm.readthedocs.io/en/latest/Parameters.html
    #https://knknkn.hatenablog.com/entry/2021/06/29/125226
    #https://zenn.dev/megane_otoko/articles/2021ad_09_optuna_optimization
    
    params = {
        "task": "train", 
        "objective": "regression",
        "boosting_type": "gbdt", 
        "metrics": {"rmse"}, 
        "learning_rate": 0.09789358290638106, 
        "num_leaves": 15, 
        "tree_learner": "feature",
        "lambda_l1": 129.61046396066124, 
        "lambda_l2": 92.07795163948921, 
        "seed": 17, 
        "max_depth": 6
        }

        #カテゴリカルデータをリストで渡す
    categorical_list = [
                        "cleaning_fee",
                        "host_identity_verified",
                        "cancellation_policy",
                        "room_type",
                        "neighbourhood",
                        "zipcode" 
                        ]
    #https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
    #https://lightgbm.readthedocs.io/en/latest/Python-API.html
    model = lgb.train(params, 
                        trains, 
                        valid_sets=valids, 
                        categorical_feature=categorical_list, 
                        num_boost_round=1000, 
                        early_stopping_rounds=50, 
                        )

    
    return model

