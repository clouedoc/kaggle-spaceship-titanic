# spaceship-titanic

> Predict which passengers are transported to an alternate dimension

This project contains my entry for the [spaceship-titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) Kaggle competition.

It currently gets a **79%** score. I'll take it :).

## Score history

| ID  | Score | Description               | Leaderboard position |
| --- | ----- | ------------------------- | -------------------- |
| 0   | 78%   | Balanced                  |                      |
| 1   | 77%   | Unbalanced                |                      |
| 2   | 79%   | Balanced, with cabin data | 1485                 |

## Improvements

- [ ] Explain what parameters matter!
- [ ] Smarter data filling
- [ ] Better feature engineering
  - [x] Exploit Cabin data
  - [ ] Exploit first name and last name
- [ ] Hyperparameter tuning for the model
- [x] Testing with and without balancing -> balanced model performs 0.1% better compared to unbalanced
