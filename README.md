# Link Prediction using metadata

Performs scientific literature keyword-keyword co-occurrence prediction based on associated metadata based features. 

## Getting Started

### Prerequisites

* [NetworkX 2.2](https://networkx.github.io/)
* [Keras](https://keras.io/)
* theano
* [statsmodel](https://www.statsmodels.org/stable/index.html)
* scipy
* numpy
* matplotlib
* pandas
* sklearn


### Files

* **link_prediction.ipynb -** End to end link prediction experiment
* **Link_Analysis.ipynb -** Data generation regarding keyword network evolution and associated characteristics
* **timeseries.ipynb -** *LSTM* based Timeseries analysis of nodal degree 
* **graphs.py -** Contains required functions to build, save and load graphs
* **utils.py -** Utility functions
* **classification.py -** Initial training-test set preparation, model training and evaluation
* **feature_selection.py -** Node level and edge level feature generation
* **versions.py -** Checks the versions of different packages

### [Dataset](https://github.com/faisal-iut/linkPrediction/tree/master/dataset)

* [Apnea dataset](https://github.com/faisal-iut/linkPrediction/blob/master/dataset/apnea-all%2C3.csv)
* [Apnea keyword lists](https://github.com/faisal-iut/linkPrediction/blob/master/dataset/apnea-distinct_keyword.csv)
* [Obesity dataset](https://github.com/faisal-iut/linkPrediction/blob/master/dataset/obesity-all%2C3.csv)
* [Obesity keyword lists](https://github.com/faisal-iut/linkPrediction/blob/master/dataset/obesity-distinct_keyword.csv) 

### Usage
1. Open [link_prediction](https://github.com/faisal-iut/linkPrediction/blob/master/link_prediction.ipynb) notebook. End-to-end link prediction experiment is done here (graph build, save, load -> training data prepare, save, load -> model training, save, evaluate -> result save, load -> figure generate, save) 
2. Experimental analysis related to keyword network evolution is done in [link_analysis](https://github.com/faisal-iut/linkPrediction/blob/master/Link_Analysis.ipynb) notebook. 
3. *LSTM* timeseries forecasting of top-3 central keywords nodal degree is done in [timeseries](https://github.com/faisal-iut/linkPrediction/blob/master/timeseries.ipynb) notebook. Then ground truth *vs* predicted value graph is generated.

## Authors

* [Fahim Faisal](https://github.com/faisal-iut)
* Dr. Nazim Choudhury

## License
[MIT](https://choosealicense.com/licenses/mit/)