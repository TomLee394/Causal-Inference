## Description
A causal inference project I worked on during my time as a data scientist for one of Canada's major banks. I was asked to measure the causal impact of the consumption of educational material (publicly available) on attrition rates for a specific product that was related to the theme of the material. Due to the nature of the treatment selection (non randomized, self-selected), the data was strictly observational (as opposed to randomized experimental) and was thus a suitable problem to solve using causal inference. Any sensitive information such as feature names, targets, and results have been removed for privacy.

## Approach
I used propensity score matching, 2 variations of meta-learners (an S learner with inverse propensity weighting and an X learner), and double/orthogonal machine learning along with cross-fitting<sup>[1]</sup> to increase confidence in my results. Individual models (i.e. treatment and outcome) were evaulated using classic ML evaluation metrics (i.e auc/pr-auc) while final CATE models were evaluated using cross-fitting [1]. What I was looking for were multiple approaches to return similar results.

## References:
1. Cross Fitting (cross validation method when the ground truth is unknowable): Jacob, Daniel. "Cross-fitting and averaging for machine learning estimation of heterogeneous treatment effects." arXiv preprint arXiv:2007.02852 (2020).
