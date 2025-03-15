# alc_attacks

A collection of attacks on anonymous data using the Anonymity Loss Coefficient (ALC). The ALC itself is based on attribute inference attacks.

ALC scores of ALC=0.5 or less can be regarded as having very strong anonymity.

## Attacks

So far there is only one attack, the simple_row_match attack. Please see the associated README.md.


## Install

Clone the repo locally.

(It goes without saying, but you should do all the following in a venv.)

You must pip install the anonymity_loss_coefficient:

`pip install git+https://github.com/yoid2000/anonymity_loss_coefficient.git`

To install the other required packages, do:

`pip install -r requirements.txt`