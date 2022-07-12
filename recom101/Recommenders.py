import numpy as np
import pandas as pd
from typing import List
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

from lightfm import LightFM
from lightfm_dataset_helper.lightfm_dataset_helper import DatasetHelper


class Implicit101:
    def __init__(self, rating_df: pd.DataFrame):
        self.rating_df = rating_df
        self.repos = None
        self.repo_ids = None

    def prepare_data(self):
        self.rating_df["user"] = self.rating_df["user"].astype("category").copy()
        self.rating_df["repo"] = self.rating_df["repo"].astype("category").copy()
        user_item_coo = coo_matrix(
            (
                np.ones(self.rating_df.shape[0]),
                (
                    self.rating_df["user"].cat.codes.copy(),
                    self.rating_df["repo"].cat.codes.copy(),
                ),
            )
        )
        self.repos = dict(enumerate(self.rating_df["repo"].cat.categories))
        self.repo_ids = {r: i for i, r in self.repos.items()}
        return user_item_coo

    def train(
        self, user_item_coo: coo_matrix, factors: int = 50, iterations: int = 500
    ):
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=0.01,
            dtype=np.float64,
            iterations=iterations,
            num_threads=0,
        )
        model.fit(user_item_coo, show_progress=True)
        return model

    def similar_repos(self, model, repo: str = "tensorflow/tensorflow"):
        return pd.DataFrame(
            {
                "user": [
                    self.repos[r] for r in model.similar_items(self.repo_ids[repo])[0]
                ],
                "repo": model.similar_items(self.repo_ids[repo])[1],
            }
        )

    def user_items(self, model, u_starred: List, factor: str = 1):
        starred_repo_ids = [self.repo_ids[s] for s in u_starred if s in self.repo_ids]
        data = [factor for _ in starred_repo_ids]
        rows = [0 for _ in starred_repo_ids]
        shape = (1, model.item_factors.shape[0])
        return coo_matrix((data, (rows, starred_repo_ids)), shape=shape).tocsr()

    def recommend(self, model, user_items: coo_matrix):
        recs = model.recommend(
            userid=0, user_items=user_items, recalculate_user=True, N=25
        )
        df = pd.DataFrame(
            {"repo": ["github.com/" + self.repos[r] for r in recs[0]], "score": recs[1]}
        )
        return df.drop("score", axis=1)

    def explain(self, model, user_items: coo_matrix, repo: str):
        _, recs, _ = model.explain(
            userid=0, user_items=user_items, itemid=self.repo_ids[repo]
        )
        df = pd.DataFrame(recs)
        df.columns = ["repo_id", "score"]
        df["repo"] = df["repo_id"].map(self.repos)
        return df[["repo", "repo_id", "score"]].drop("repo_id", axis=1)


class LightFM101:
    def __init__(self, rating_df: pd.DataFrame):
        self.rating_df = rating_df
        self.items_column = "repo"
        self.user_column = "user"
        self.ratings_column = "rating"

        self.items_feature_columns = [
            "event_cnt",
            "user_uniq_cnt",
        ]
        self.user_features_columns = ["repo_cnt", "repo_uniq_cnt"]

    def prepare_data(self):
        dataset_helper_instance = DatasetHelper(
            users_dataframe=self.rating_df[
                [self.user_column] + self.user_features_columns
            ],
            items_dataframe=self.rating_df[
                [self.items_column] + self.items_feature_columns
            ],
            interactions_dataframe=self.rating_df[
                [self.user_column, self.items_column, self.ratings_column]
            ],
            item_id_column=self.items_column,
            items_feature_columns=self.items_feature_columns,
            user_id_column=self.user_column,
            user_features_columns=self.user_features_columns,
            interaction_column=self.ratings_column,
            clean_unknown_interactions=True,
        )

        dataset_helper_instance.routine()
        return dataset_helper_instance

    def train(self, dataset_helper_instance, factors: int = 50, iterations: int = 500):
        model = LightFM(no_components=factors, loss="warp", k=25)
        model.fit(
            interactions=dataset_helper_instance.interactions,
            sample_weight=dataset_helper_instance.weights,
            item_features=dataset_helper_instance.item_features_list,
            user_features=dataset_helper_instance.user_features_list,
            verbose=True,
            epochs=iterations,
            num_threads=12,
        )
        return model

    def recommend(
        self,
        dataset_helper_instance,
        model,
        u_starred: List,
        user: str = "recom_client",
        top_n: int = 25,
    ):
        item_ids = dataset_helper_instance.get_item_id_mapping().values()
        tmp_user_id = dataset_helper_instance.get_user_id_mapping().get(user)
        scores = model.predict(
            user_ids=tmp_user_id,
            item_ids=list(item_ids),
            user_features=dataset_helper_instance.user_features_list,
            item_features=dataset_helper_instance.item_features_list,
        )
        scores_df = pd.DataFrame(
            {
                "repo": dataset_helper_instance.get_item_id_mapping().keys(),
                "score": np.around(np.array(scores, dtype=np.float64), 3),
            }
        )

        scores_df = scores_df[lambda df: np.logical_not(df.repo.isin(u_starred))].copy()
        scores_df["repo"] = scores_df["repo"].apply(lambda x: f"github.com/{x}")
        scores_df = (
            scores_df.sort_values("score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        return scores_df.drop('score', axis=1).round(3)
