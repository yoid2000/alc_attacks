# Best Row Match attack

This attack is designed to work with microdata (aka synthetic data). The attack is an attribute inference attack: the attacker knows from information about a person in the anonymized data, finds a row that "best matches" that person, and the infers unknown attributes.

We use the Anonymity Loss Coefficient to measure anonymity. The ALC compares the attack inference with a baseline inference that uses an ML model to predict the unknown attribute given the known attributes.

A unique feature of our attack is that it can handle the case where multiple synthetic datasets have been released. Essentially, it runs the attack on all of the synthetic datasets, and selects the most common answer as the best answer.

## Example

See `alc_attacks/examples/best_row_match` for example code on how to use `brm_attack.py`. Note that `alc_attacks/examples/best_row_match/run_brm_attack.py` can be used stand-alone as a complete attack. 

## Limitations

We currently don't recognize datetime columns as datetime columns.

We only input csv files. Should add parquet to this in the future.