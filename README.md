# VK_Scoring_flask

XGBoost binary classifier with ROC AUC 0.74.

## Setup guide

```
* Create a python 3.x venv
* $ pip install -r requirements.txt
* run server.py (change the url if not running locally)
* change the example query in request.py (reroute to receive posts from some other machine if needed)
* run request.py in a separate terminal for inference (change the url to match 'url:port/api' if not running locally)
```

The resulting float is a probability of VK profile to be prorussian and dangerous (the higher, the more likely)

## Improvements

ML: Try different approaches in feature engineering
Data engineering: create a docker container

@CyberFI 2022