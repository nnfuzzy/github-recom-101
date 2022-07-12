import pandas as pd
import datetime
import streamlit as st
from loguru import logger
from recom101.Config import AppConfig
from recom101.DataPrep import DataPrep
from recom101.Recommenders import Implicit101
from recom101.Recommenders import LightFM101


main_title = "Github recommendations"
st.set_page_config(page_title=main_title)
st.header("Recommendation for github repos based on your interested repos")
st.markdown("1. Don't overstretch the parameters (e.g. time window in the demo)")
st.markdown("2. Don't forget to upload some repos (your preferences) ")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css(".streamlit/st.css")


with st.sidebar.form(key="form"):
    start_button = st.form_submit_button(label="Start recommendation")
    st.markdown("### Recommender settings:")

    uploaded_file = st.file_uploader(
        "Upload your preferences:",
        accept_multiple_files=False,
        type="json",
        help="""Upload a simple json file with a list of repos you like \n
        {"repo": ["tensorflow/recommenders",
                  "tensorflow/probability"]
                  } """,
    )

    iterations = st.slider(
        "Iterations:",
        1,
        2500,
        value=100,
        step=25,
        help="Iterations - collaborative filtering model",
    )

    min_items = st.slider(
        "Min unique repos:",
        1,
        20,
        value=5,
        step=2,
        help="Min threshold - unique repos by users",
    )
    max_items = st.slider(
        "Max unique repos:",
        5,
        100,
        value=50,
        step=5,
        help="Min threshold - unique repos by users",
    )

    top_items = st.slider(
        "Top repos:",
        0,
        10_000,
        value=250,
        step=250,
        help="Use only top n repos with highest unique user volume",
    )

    unique_user_threshold = st.slider(
        "Repo unique user threshold:",
        0.2,
        1.00,
        value=0.90,
        step=0.01,
        help="Repo unique user threshold",
    )


    possible_algo_libs = {v: i for i, v in enumerate(["implicit-als", "lightfm-warp"])}
    sel_algo_lib = st.selectbox(
        "Algorithm:",
        list(possible_algo_libs.keys()),
        help="Used algorithm",
    )

    possible_factors = {v: i for i, v in enumerate([48, 60, 72, 84])}
    sel_factors = st.selectbox(
        "Latent factors:",
        list(possible_factors.keys()),
        index=2,
        help="Latent factors - collaborative filtering",
    )

    st.write("Datetime window:")
    min_date = datetime.datetime(2022, 1, 1)
    max_date = datetime.datetime(2022, 6, 30)
    start_date = st.date_input(
        "Start date",
        value=datetime.datetime(2022, 6, 1),
        min_value=min_date,
        max_value=max_date,
        help="Time window start dt",
    ).strftime("%Y-%m-%d")

    end_date = st.date_input(
        "End date",
        value=datetime.datetime(2022, 6, 8),
        min_value=min_date,
        max_value=max_date,
        help="Time window end dt",
    ).strftime("%Y-%m-%d")

    app_config = AppConfig()
    data_prep = DataPrep(
        data_path=app_config.data_path, github_token=app_config.github_token
    )

    print(uploaded_file)
if start_button:
    try:
        if uploaded_file:
            client_repos_df, client_repos_list = data_prep.prepare_user_data(
                json_file_object=uploaded_file, min_dt=start_date
            )
        else:
            st.error("You need to upload some preferences!")
            st.stop()

        t0 = datetime.datetime.now()
        github_df = data_prep.load_data(polars=True)
        # st.write(df.head(10))
        filtered_df = data_prep.filter_data_polars(
            github_df=github_df,
            client_repos_df=client_repos_df,
            min_dt=start_date,
            max_dt=end_date,
            min_items=int(min_items),
            max_items=int(max_items),
            top_items=top_items,
            unique_user_threshold=float(unique_user_threshold)
        )
        st.info(f"Data input has {len(filtered_df)} rows with limited repos")
        t1 = datetime.datetime.now()
        print((t1 - t0).total_seconds())
        st.markdown("# Recommendations")

        if sel_algo_lib == "implicit-als":
            implcit101 = Implicit101(rating_df=filtered_df)
            coo_matrix = implcit101.prepare_data()
            model = implcit101.train(
                user_item_coo=coo_matrix,
                factors=possible_factors[sel_factors],
                iterations=iterations,
            )

            # u_starred = data_prep.get_user_repo_starred()
            user_items = implcit101.user_items(model, u_starred=client_repos_list)
            logger.info(f"user_items: {user_items}")
            recom_df = implcit101.recommend(model, user_items=user_items)
            logger.info(f"recoms: {recom_df}")
            # st.markdown("## Top Recommendations")
            st.dataframe(recom_df)

        elif sel_algo_lib == "lightfm-warp":
            lightfm101 = LightFM101(rating_df=filtered_df)
            ds = lightfm101.prepare_data()
            model = lightfm101.train(
                ds, factors=possible_factors[sel_factors], iterations=iterations
            )
            recom_df = lightfm101.recommend(ds, model, client_repos_list)
            st.dataframe(recom_df)
    except Exception as e:
        st.error(e)
    finally:
        st.stop()
