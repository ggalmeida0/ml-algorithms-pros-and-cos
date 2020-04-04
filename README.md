# Machine Learning Algorithms - Strenghts and Weaknesses



## K-Nearest Neighbors (Classification and Regression):
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

## Linear Models (Classification and Regression):
### Meaningful Parameters:
- **Regularization parameter**,simply a parameter that determines how much complexity the model is allowed to have. But more in dept is a parameter that determines how big the range of allowed values is for w in the formula: \( hat{y} \)


## Credits:
- *Introduction to Machine Learning with Python by Andreas C. Muller & Sarah Guido*
