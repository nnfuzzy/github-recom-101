import pandas as pd
from loguru import logger
# your gcp project
project_id = None


""" Hint, don't do 'Select * from table' things in big query if you don't need to."""

rs = []
for i in range(1, 8):
    logger.info(f"Query month {i}")
    query = f"""SELECT actor.login AS user, repo.name AS repo, created_at AS timestamp from githubarchive.month.20220{i} where type = 'WatchEvent'"""
    tmp_df = pd.io.gbq.read_gbq(query, index_col="timestamp", dialect="standard", project_id=project_id)
    rs.append(tmp_df)

dfrs = pd.concat(rs)
dfrs.to_csv("github_archive_2022q1q2.csv", sep=";")