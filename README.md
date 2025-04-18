# alc_attacks

A collection of attacks on anonymous data using the Anonymity Loss Coefficient (ALC). The ALC itself is based on attribute inference attacks.

ALC scores of ALC=0.5 or less can be regarded as having very strong anonymity.

## Attacks

So far there is only one attack, the best_row_match attack. Please see the associated README.md.


## Install

If you only wish to run a best row match attack, easiest to simply copy `examples/best_row_match/run_brm_attack.py`, do `pip install alc_attacks`, and run the code according to the directions at `examples/best_row_match/README.md`.


## Dev

To push to pypi.org:

remove the `dist/` directory

Update the version in `setup.py`

`python setup.py sdist bdist_wheel`

`twine upload dist/*`