# Data Mining Midterm Project — BPN & Decision Tree Classification

This project implements and compares two classical machine learning classification methods: Back-Propagation Neural Networks (BPN) and Decision Trees. Both models are built from scratch using Python and evaluated against three real-world sports datasets.

The BPN is designed to fulfill the core requirements of a supervised learning classifier: it supports learning through gradient descent, recalling predictions on unseen data, and configurable stop conditions based on training error or number of iterations. Different hidden layer architectures and learning rates are tested on each dataset to analyze their effect on model performance.

The Decision Tree classifier is trained on the same datasets using the same preprocessing pipeline, allowing for a direct and fair comparison of accuracy, precision, recall, and F1-score between both methods.

Three datasets are used to validate the models. The NFL Combine dataset (1996–2022) contains physical measurements from pre-draft evaluations, where the goal is to classify players into position groups such as Skill, Lineman, D-Line, Linebacker, Secondary, and Special Teams. The FIFA 20 player dataset includes in-game skill attributes for thousands of players worldwide, and the classification task is to predict a player's preferred foot (Left or Right). The NBA historical stats dataset spans several decades of player performance data, and the model predicts whether a player was selected as an All-Star during a given season based on their stats such as points, rebounds, assists, steals, blocks, and minutes played.

Results from both Python implementations are compared against Weka, a well-known data mining workbench, to validate consistency across tools and provide an additional benchmark for model evaluation.

## Datasets
- **NFL Combine** — Physical measurements to classify player position group
- **FIFA 20** — Player skill attributes to predict preferred foot (Left / Right)
- **NBA All-Star** — Season stats to predict All-Star selection (Yes / No)

## Methods
- Back-Propagation Neural Network (BPN) — `sklearn.neural_network.MLPClassifier`
- Decision Tree — `sklearn.tree.DecisionTreeClassifier`

## Project Structure
| File | Description |
|---|---|
| `bpn_nfl.ipynb` | BPN on NFL Combine dataset |
| `tree_nfl.ipynb` | Decision Tree on NFL Combine dataset |
| `bpn_fifa20.ipynb` | BPN on FIFA 20 dataset |
| `tree_fifa20.ipynb` | Decision Tree on FIFA 20 dataset |
| `bpn_nba.ipynb` | BPN on NBA All-Star dataset |
| `tree_nba.ipynb` | Decision Tree on NBA All-Star dataset |
