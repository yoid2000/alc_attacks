# Simple Row Match attack

This attack is designed to work with microdata (aka synthetic data). The attack is an attribute inference attack: the attacker knows from information about a person in the anonymized data, finds a row that "best matches" that person, and the infers unknown attributes.

The attack itself is similar to the Anonymeter attack in how it finds the best row (see the paper at https://arxiv.org/pdf/2211.10459), and indeed we borrow some of their code.

We use the Anonymity Loss Coefficient to measure anonymity. The ALC compares the attack inference with a baseline inference that uses an ML model to predict the unknown attribute given the known attributes.

A unique feature of our attack is that it can handle the case where multiple synthetic datasets have been released. Essentially, it runs the attack on all of the synthetic datasets, and selects the most common answer as the best answer.

## To run

Navigate to simple_row_match.

Run `python srm_attack.py attack /path/to/attack_directory`

## Setup

`srm_attack.y` expects to find everything it needs in `attack_directory/inputs`. Specifically, the following should be placed there:

* `original.csv`: Contains the original data, minus randomly selected rows that are set aside as controls for computing the baseline.

* `control.csv`: This file contains randomly selected rows from `original.csv`. We recommend 1000 rows.

* `synthetic_files`: This is a directory containing one or more synthetic datasets generated from `original.csv`.

All of the csv files must have the same columns.

Note that the larger `original.csv` is, the longer it'll take to run the tests. Therefore it might be a good idea to limit `original.csv` (and subsequently the synthetic dataset) to 10k-20k or so rows.

See the `tests` file for example of the setup.


## Results

`srm_attack.py` creates a directory `results` under `attack_diretory`. `results` contains three files:

* `summary_raw.csv`: This contains the results of every individual prediction, both baseline and attack.
* `summary_secret.csv`: This contains a row for every secret (unknown) attribute being predicted. It gives the precision for all predictions for both baseline and attack, and computes the ALC score.
* * `summary_secret_known.csv`: This contains a row for every combination of secret and known attributes.  It gives the precision for all predictions for both baseline and attack, and computes the ALC score.

## Operation

`srm_attack.py` does the following:

It identifies the categorical columns to be used as unknown (secret) attributes.

For each secret:

* It discovers sets of known attributes that uniquely define each row of the data.
* It makes baseline and attack predictions of the secret data.

`srm_attack.py` loops through the sets of unknown attributes. In each loop, it runs an attack (a set of predictions) on every unknown attribute. After each such attack, it updates the three results files. In this fashion, the results files continuously receive more results data as `srm_attack.py` runs.

Note that `srm_attack.py` can take a long time to run (many hours), so it is good to just let it go while the results files build up. 



## Limitations

We currently only run attacks on categorical columns as the unknown column.

We currently don't recognize datetime columns as datetime columns.

We only input csv files. Should add parquet to this in the future.