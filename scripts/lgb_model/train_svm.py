# -*- coding: utf-8 -*-
# @shiweitong 2024/4/12
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)
from spark_learning.CatEncoder import CatEncoder
from spark_learning.utils.models import save_model
from config import X, y


def transform_input_raw(dataset: pd.DataFrame, encoder: (str, CatEncoder), with_label=True, update_encoder=False):
    cat_features = [
        'entity_country', 'entity_region', 'genre', 'sub_genre', 'tag_list',
        'developer_id', 'developer', 'publisher_id', 'publisher',
    ]
    if not isinstance(encoder, CatEncoder) and (update_encoder or not os.path.exists(encoder)):  # 换新的encoder
        _encoder = CatEncoder()
        for fea in cat_features:
            dataset[fea] = _encoder.fit_transform(
                fea,
                # dataset[fea]
                dataset[fea].apply(lambda x: (x.split("|")[0] if "|" in x else x) if isinstance(x, str) else x)
            )
        _encoder.save(encoder)
    else:  # 用之前的encoder
        encoder = CatEncoder.load(encoder) if isinstance(encoder, str) else encoder
        for fea in cat_features:
            dataset[fea] = encoder.transform(
                fea,
                # dataset[fea]
                dataset[fea].apply(lambda x: (x.split("|")[0] if "|" in x else x) if isinstance(x, str) else x)
            )

    time_features = [
        "date",
    ]
    for fea in time_features:
        dataset[fea] = (pd.to_datetime(dataset['release_time']).dt.tz_localize(None) - pd.to_datetime(dataset[fea])
                        ).dt.days

    return dataset


def train_pheat_demo():
    logging.info("loading training data")
    dataset = pd.read_excel("data/train.xlsx")

    # 把非数字信息转换为序号标签
    dataset = transform_input_raw(dataset, "cat_encoder.dill", update_encoder=True)

    dataset['average_acu'] = dataset.groupby('edition_id')['EA_acu'].transform('mean')
    dataset['date'] = pd.to_datetime(dataset['date'], format='mixed')
    dataset = dataset.sort_values(by='date')
    dataset = dataset.reset_index(drop=True)

    main_dataset = dataset[~dataset["wishlist_rank"].isna()]  # 有ranking的数据行
    main_games = main_dataset["edition_id"].unique()  # 有ranking的游戏
    supplement_dataset = dataset[dataset["wishlist_rank"].isna()]  # 无ranking的数据行
    supplement_games = supplement_dataset["edition_id"].unique()  # 无ranking的游戏

    logging.info(f"main games: {len(main_games)}; supplement games: {len(supplement_games)}")
    games_train, games_test = train_test_split(
        np.concatenate([main_games, supplement_games]),
        stratify=[0] * len(main_games) + [1] * len(supplement_games),  # 保证train test里面有无ranking的游戏比例是一样的
        test_size=0.1,
        random_state=43,
    )
    logging.info(f"games for train: {len(games_train)}, games for test: {len(games_test)}")

    train_ds = dataset[dataset["edition_id"].isin(games_train)]  # train test的游戏都加上数据
    eval_ds = dataset[dataset["edition_id"].isin(games_test)]

    logging.info(f"Train: {len(train_ds)}, Eval {len(eval_ds)}")

    x_train, y_train = train_ds[X], train_ds[y]  # 预测pheat
    x_eval, y_eval = eval_ds[X], eval_ds[y]

    categorical_features = ['developer_id', 'developer', 'publisher_id', 'publisher', 'genre', 'sub_genre', 'tag_list']
    numerical_features = ['wishlist_rank', 'EA_revenue', 'EA_pcu', 'EA_acu', 'average_acu']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    model = make_pipeline(preprocessor, SVR(kernel='linear'))

    model.fit(x_train, y_train)

    # ------------------ Evaluation ----------------
    train_pred = model.predict(x_train)
    eval_pred = model.predict(x_eval)

    logging.info({
        "ndcg_score_train": ndcg_score(np.expand_dims(y_train, 0), np.expand_dims(train_pred, 0)),
        "ndcg_score_eval": ndcg_score(np.expand_dims(y_eval, 0), np.expand_dims(eval_pred, 0)),
    })

    save_model(model, "lgb.dill")


if __name__ == '__main__':
    from spark_learning.utils import config_logging

    config_logging()

    train_pheat_demo()
