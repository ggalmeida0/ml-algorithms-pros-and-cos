# Machine Learning Algorithms - Strenghts and Weaknesses
A summarization of the pros and cons of the machine learning algorithms presented in the book Introduction to Machine Learning with Python by Andreas C. Muller & Sarah Guido


## K-Nearest Neighbors (classification and regression):
### Meaninful Parameters:
- **Number of neighbors**, the number of samples closest to the point we're predicting.
  - How to tune it: *Low numbers ofter work best but experimentation is required*
- **Measure of Distance**, the method use to measure the distance of point we're predicting and the neighbors.
  - How to tune it: *Euclidean Distance will generally perform well*
  
### Strengths:
- **The simplest ML algorithm** - You can use the KNN to test the data quickly and see if a more complex aprouch is required
- **Little tuning required** - The KNN can preform well with little parameter tunning.

### Weaknesses:
- **Can be very slow** - When the dataset is very big in number of samples or features this algorithm takes a long time to make predictions.
- **Performs pourly with sparse datasets** - When there are a lot of data points equal to zero theres a significant performance drop in KNNs.

## Linear Models (Linear Regression, Support Vector Machine, Ridge, Logistic Regression etc):
### Meaningful Parameters:
- **Regularization parameter**,simply a parameter that determines how much complexity the model is allowed to have. But more in dept is a parameter that determines how big the range of allowed values is for _w_ in the formula: ![equation](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Chat%7By%7D%20%3D%20w%5Bi%5D%20%5Ccdot%20x%5Bi%5D%20&plus;%20b)
- **L1 or L2 regularization**, these are regularization criteria that helps the model to not overfit. Generally if you believe that only a few features of the data is imporatant then use L1, otherwise use L2.

### Strengths:
- **Very Fast** - Linear models have very fast training time and prediction time.
- **Scalable** - Linear models are good at scaling to huge datsets

### Weaknesses:
- **Perform badly on lower dimensions** Linear models tend to perform well on datasets with lots of features compared to the sample size but it performs poorly on lower dimensional data.

## Naive Bayes Classifiers (only classification):
### Meaningful Parameters:
- **Alpha**, controls the complexity of the model, the higher the alpha the lower the complexity.

### Strengths:
- **Extremely fast** - NB classifiers are even faster than linear classifiers both on training and predicting, thus they can be used to get faster results where even the linear models take a long time.
- **Robust parameter settings** - NB classifiers can perform relatevely well even the parameter alpha isn't fine tuned.
- **Scalable** - Just like linear models NB classifiers are very scallable to big datasets.

### Weaknesses:
- **Assume independence** - The naive bayes classifier assumes that every variable is independent of each other, that's why the name is "naive". Thus it performs badly on datasets that have a lot of variables that depend on each other, a example would be a picture, where every pixel is dependent on each other to form a image.

## Decision Trees(regression and classification):
### Meaningful Parameters:
- **Pruning parameters**, a parameter that reflects on the complexity of the tree by reducing the number of nodes in the tree.

### Strengths:
- **Simply intrepreted** - Decision trees are one of the most explainable machine learning algorithms to non techinical people.
- **Scalable** - Decision trees are completely unvariant scaling, so it can be applied on very big datasets.

### Weaknesses:
- **Overfiting** - Decision trees are very easy to overfit the training data, even when you ajust the pruning parameter.

## Credit:
- *Introduction to Machine Learning with Python by Andreas C. Muller & Sarah Guido*
