# Best Row Match attack

This attack is designed to work with microdata (aka synthetic data). The attack is an attribute inference attack: the attacker knows from information about a person in the anonymized data, finds a row that "best matches" that person, and the infers unknown attributes.

The attack itself is similar to the Anonymeter attack in how it finds the best row (see the paper at https://arxiv.org/pdf/2211.10459), and indeed we borrow some of their code.

We use the Anonymity Loss Coefficient to measure anonymity. The ALC compares the attack inference with a baseline inference that uses an ML model to predict the unknown attribute given the known attributes.

A unique feature of our attack is that it can handle the case where multiple synthetic datasets have been released. Essentially, it runs the attack on all of the synthetic datasets, and selects the most common answer as the best answer.

## To run

Navigate to best_row_match.

Run `python brm_attack.py attack /path/to/attack_directory`

## Setup

`brm_attack.y` expects to find everything it needs in `attack_directory/inputs`. Specifically, the following should be placed there:

* `original.csv`: Contains the original data, minus randomly selected rows that are set aside as controls for computing the baseline.

* `control.csv`: This file contains randomly selected rows from `original.csv`. We recommend 1000 rows.

* `synthetic_files`: This is a directory containing one or more synthetic datasets generated from the original data. Note that this data can come from all of the original data (both `original.csv` and `control.csv`). However, it is also ok if `synthetic_files` comes only from `original.csv`.

All of the csv files must have the same columns.

Note that the larger the synthetic datasets are, the longer it'll take to run the tests. Therefore it might be a good idea to limit the synthetic datasets (and likewise the original data from which they were derived) to 10k-20k or so rows.

See the `tests` file for example of the setup.


## Results

`brm_attack.py` creates a directory `results` under `attack_diretory`. `results` contains these files:

* `summary_raw.csv`: This contains the results of every individual prediction, both baseline and attack.
* `summary_secret.csv`: This contains data every secret (unknown) attribute being predicted. It gives the precision, recall, ALC score, and confidence intervals for a variety of different recall values.
* `summary_secret_known.csv`: This contains a row for every combination of secret and known attributes. It gives the precision, recall, ALC score, and confidence intervals for a variety of different recall values.
* `summary.txt`: Gives a text summary of the results, included an anonymity grade ranging from VERY STRONG to VERY POOR.
* `alc_plot.png`: A plot summarizing the ALC scores for each set of secret attributes.
* `alc_plot_best.png`: A plot summarizing the ALC scores for the highest ALC score among a given set of secret attribute and known attributes.
* `alc_plot_prec.png`: A scatterplot showing the ALC score and attack precision for each set of secret and known attributes.
* `alc_plot_prec_best.png`: A scatterplot showing the highest ALC score and attack precision for each set of secret and known attributes.

ALC scores of ALC=0.5 or less can be regarded as having very strong anonymity.

## Operation

`brm_attack.py` does the following:

It identifies the categorical columns to be used as unknown (secret) attributes.

For each secret:

* It runs an attack assuming that all other columns are known attributes.
* It discovers smaller sets of known attributes that uniquely define each row of the data.
* It selects a set of secret values where each value constitutes fewer than 60% of the rows, but more than 0.05% of the rows. We avoid very common values because they tend not to be sensitive. We avoid very rare values because most predictions tend to be False, and an occasional random correct prediction can skew the results.
* After shuffling the target rows, it steps through the rows for both baseline and synthetic data attacks, making a prediction for each row. It quits when either the confidence bounds are with 10% of the precision for either the baseline or the attacks, and when at least 5 True predictions have been made for both baseline and attack.
* When making a prediction, if there are multiple synthetic data files, it first makes a prediction for each data file, and then uses the most common prediction as the prediction. It also assigns a confidence value to the prediction, depending on how many of the individual predictions constitute the most common prediction.
* It updates the three results files. In this fashion, the results files continuously receive more results data as `brm_attack.py` runs.

Note that `brm_attack.py` can take a long time to run (many hours), so it is good to just let it go while the results files build up. 

## Limitations

We currently don't recognize datetime columns as datetime columns.

We only input csv files. Should add parquet to this in the future.