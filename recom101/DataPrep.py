import os
import pandas as pd
import polars as pl
import numpy as np
import streamlit as st
from github import Github


class DataPrep:
    def __init__(
        self,
        project_id=None,
        github_user=None,
        github_token=None,
        data_path=None,
        filename: str = "github_archive_2022q1q2.csv",
        item_name: str = "repo",
        user_name: str = "user",
    ):
        self.project_id = project_id
        self.github_user = github_user
        self.github_token = github_token
        self.data_path = data_path
        self.filename = filename
        self.file_path = os.path.join(data_path, self.filename)
        self.query = """SELECT actor.login AS user, repo.name AS item, 
        created_at AS timestamp from githubarchive.month.202206 where type = 'WatchEvent' """
        self.item_name = item_name
        self.user_name = user_name

    @st.cache
    def load_data(self, polars=True):
        if self.project_id is None and self.data_path and self.filename and not polars:
            return pd.read_csv(self.file_path, sep=";", index_col=None)
        elif self.project_id is None and self.data_path and self.filename and polars:
            return pl.scan_csv(self.file_path, sep=";")
        elif self.project_id:
            return pd.io.gbq.read_gbq(
                self.query,
                index_col="timestamp",
                dialect="standard",
                project_id=self.project_id,
            )

    def prepare_user_data(
        self, json_file_object, min_dt: str = "2022-06-01"
    ):
        client_repos_df = pd.read_json(json_file_object)
        client_repos_df["user"] = "recom_client"
        client_repos_df["timestamp"] = min_dt
        client_repos_df = client_repos_df[["timestamp", "user", "repo"]]
        return client_repos_df, client_repos_df.repo.unique().tolist()

    def filter_data_polars(
        self,
        github_df: pl.DataFrame,
        client_repos_df: pd.DataFrame = None,
        min_items: int = 5,
        max_items: int = 50,
        min_dt: str = "2022-06-01",
        max_dt: str = "2022-06-02",
        top_items: int = 1_000,
        unique_user_threshold: float = 0.9,
    ):

        github_df = github_df.filter(
            pl.col("timestamp").is_between(min_dt, max_dt)
        ).collect()

        if len(client_repos_df) > 0:
            client_repos_df = pl.DataFrame(client_repos_df)
            github_df = pl.concat([github_df, client_repos_df])

        st.info(f"""After filtering date window  we have:""")
        st.info(f"""{len(github_df)} events """)
        st.info(f"""{github_df["user"].n_unique()} users""")
        st.info(f"""{github_df["repo"].n_unique()} repos""")

        # user features
        user_features_df = github_df.groupby("user").agg(
            [
                pl.count("timestamp").alias("repo_cnt"),
                pl.n_unique("repo").alias("repo_uniq_cnt"),
            ]
        )

        user_features_df = user_features_df.sort("repo_cnt", reverse=True).filter(
            (pl.col("repo_uniq_cnt").is_between(min_items, max_items))
            | (pl.col("user") == "recom_client")
        )

        st.info(
            f"""After filtering users we have: {user_features_df["user"].n_unique()} users"""
        )

        # item features
        item_features_df = github_df.groupby("repo").agg(
            [
                pl.count("timestamp").alias("event_cnt"),
                pl.n_unique("user").alias("user_uniq_cnt"),
            ]
        )
        item_features_df = (
            item_features_df.sort("event_cnt", reverse=True)
            .filter(
                (
                    (
                        (
                            pl.col("user_uniq_cnt")
                            > (
                                pl.col("event_cnt").cast(pl.Float64)
                                * unique_user_threshold
                            )
                        )
                        | (pl.col("repo").is_in(list(client_repos_df.select("repo"))))
                    )
                )
            )
            .sort("user_uniq_cnt", reverse=True)
        )

        st.info(
            f"""After filtering repos we have: {item_features_df["repo"].n_unique()} repos"""
        )

        rating_df = github_df.groupby(["user", "repo"]).agg(
            [pl.count("timestamp").alias("rating")]
        )

        rating_merged_df = (
            rating_df.join(item_features_df.head(top_items), on=["repo"])
            .join(user_features_df, on=["user"])
            .to_pandas()
        )
        return rating_merged_df

    @st.cache
    def get_user_repo_starred(self, user: str = "nnfuzzy"):
        g = Github(self.github_token)
        return [repo.full_name for repo in g.get_user().get_starred()]
