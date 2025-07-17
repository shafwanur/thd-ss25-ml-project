# VPN Traffic Classification using ISCXVPN2016 Dataset

## Project Overview

This project accompanies our 4th semester Machine Learning course at Deggendorf Institute of Technology. It analyzes the **ISCX VPN-nonVPN 2016 dataset** to classify internet traffic into:
- **VPN vs non-VPN**
- **One of seven application categories**: Browsing, Email, Chat, Streaming, File Transfer, VoIP, and P2P.

We applied stastistical learning techniques to do a machine learning evaluation on the dataset using scikit-learn.

**Dataset**: [ISCX VPN-nonVPN 2016](https://www.unb.ca/cic/datasets/vpn.html)  
**Original Paper**: [A Reliable Labeled Dataset for VPN Traffic Identification](https://www.scitepress.org/Papers/2016/57407/57407.pdf)

---

## Models Used

We implemented three classification models:

1. **k-Nearest Neighbors (kNN)**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

The models were trained and tested using a standard train/test split. We used randomized grid search to find the best hyperparameters. Despite issues in the dataset (discussed below), we evaluated the models' performance to understand the limitations and potential of the models based on our dataset.

## Performance Comparison

| Model             | Train Accuracy | Test Accuracy |
|-------------------|----------------|---------------|
| kNN              | 100%           | 77.62%           |
| Decision Tree     | 99.76%            | 85.67%           |
| Random Forest     | 100%            | 88.91%           |

---

## Adressing the clear overfitting

- **Data Leakage**: The dataset includes overlapping timestamps that lead to information leaking throughout the dataset as a whole. This probably explains why we basically have a 100% train accuracy across all models. It was impossible to seperate the train/test split so that information doesn't flow from one sample to the other. This of course messes with the predictions and the generalisation of the models.
- **Lack of Metadata**: Crucial information like flow direction or session context was not well-defined and were not made public by the creators of the dataset. In an attempt to fix these issues, we thoroughly reviewed the [original paper](https://www.scitepress.org/Papers/2016/57407/57407.pdf), but ultimately a lack of transparency on the origin of the dataset hindered our ability to fix the data leakage issue. You can see the data exploration and how we discovered we have issues in the dataset in the [features.ipynb](notebooks/exploration/features.ipynb) notebook. 

---

## Lessons Learned

Nonetheless, we performed a machine learning analysis on the dataset and gained valuable insights by doing so:
- We learned how to perform dataset evaluation and quality assessments.
- We encountered common ML pitfalls, like data leakage, and learned how one could and should avoid them them.
- We carried out comparative model evaluation with simple vs more advanced classifiers.
- We performed thorough hyperparameter optimisation using various techniques to achieve the best validation scores. 