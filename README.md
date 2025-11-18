# College Football Play Predictor

## Scrapers
* All 2024 Power 4 CFB ESPN game Ids were scraped using **/Scrapers/GameIdScraper.py**
* All plays from these games were obtained from the public ESPN API using **/Scrapers/PlayScraper.py**

## Models
* 6 different models have been trained and evaluated
    * Conditional Inference (/Models/ConditionalInference.py)
    * Gradiant Boosting (/Models/GradiantBoosting.py)
    * K Nearest Neighbor (/Models/KNearestNeighbor.py)
    * Logistic Regression (/Models/LogisticRegression.py)
    * Naive Bayes (/Models/NaiveBayes.py)
    * Random Forest (/Models/RandomForest.py)


## Installation and Running

### Python Dependencies
* Install required Python packages:
```
> pip install pandas scikit-learn joblib numpy requests beautifulsoup4
```

### R Dependencies (Only for Conditional Inference)
* The Conditional Inference model requires R and the `party` package. A Fortran compiler must also be installed.
```
> pip install rpy2
> R
> install.packages("party")
> quit()
```

### Running a Model
* Each model can be run with:
```
> python Models/ModelName.py
```