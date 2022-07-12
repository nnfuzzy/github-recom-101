# Github Recommender 101

An experimental recommender inside of streamlit to keep updated in new developments in terms of my preferences.

<video src="https://user-images.githubusercontent.com/687670/178553753-ed82fb41-622c-475d-aa1e-3f692a07e3ee.mp4" controls="controls" style="max-width: 730px;"> </video>

## Data

1. Choose and prepare data (via gcp you can access github data related to public projects).

## Recommenders

1. Setup different recommenders
    * implict (done)
    * LightFm (done)
    * tensorflow
    * fastapi (tricky in terms of implict issues, because it expects ratings and fill 0 value combos will explode the data)

## Streamlit

1. Streamlit is an ideal tool to visualize data-science stuff quickly and using as background for discussion or a low cost prototyping environment.

#### Running

```
DATA_PATH="data/" streamlit run recom_app.py
``` 


## Dockerfile

1. To use it in proper way you can build your docker image.
2. Github action makes it easy to import the image to AWS ECR for further usage in EC2, Fargate or AppRunner.

#### Running local
  
```
docker build -t recom101 . 
docker run --rm recom101      
```

## Possible future

* Adding more algorithms
* Make everything more configurable with the json file
* Make it more generic for different data use cases

## Credits 

Public community and ...

* [streamlit](https://github.com/streamlit/streamlit)
* [tensorflow-recommenders](https://github.com/tensorflow/recommenders)
* [loguru](https://github.com/Delgan/loguru)
* [pandas](https://github.com/pandas-dev/pandas)
* [lightfm](https://github.com/lyst/lightfm)
* [lightfm-dataset-helper](https://github.com/Med-ELOMARI/LightFM-Dataset-Helper)
* [polars](https://github.com/pola-rs/polars/)
