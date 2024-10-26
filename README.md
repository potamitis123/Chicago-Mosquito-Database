Code for the paper 'Unveiling Mosquito Patterns in Chicago (2007-2024): A Data Analytics and Machine Learning Study, by Dr. Ilyas Potamitis

original_cardal: the submissions for the winning solution of https://www.kaggle.com/c/predict-west-nile-virus

Chicago_WNV: The Chicago Mosquito database (2007-2024). Taken from https://data.cityofchicago.org/Health-Human-Services/West-Nile-Virus-WNV-Mosquito-Test-Results/jqe8-8r6s/about_data

Chicago_WNV_figures.py: Reproduces all figures in the paper

parse_Chicago_WNV.py: The refactored Cardal approach refactored for Python 3.7 and leederboard fitting practices removed

Our contribution
bivariate.py: Fit a bivariate Normal on log(NumMosquitos+1) and WnvPresent Dates and derive the probability of an infected batch

bivariate_with_trap_bias.py: Fit a bivariate Normal and also apply a statistical significance test on traps that leads to a weight applied to the prediction prob.

Tree_based_Chicago_WNV.py: Apply tree-based algorithms on the dataset


