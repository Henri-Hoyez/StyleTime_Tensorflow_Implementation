#/bin/sh

# ENERGY DATASET.
## IN SMAPLE DATASET
screen -d -m python paper_style_transfer.py 'data/energy/preprocessed/in_sample_train.npy' 'data/energy/preprocessed/style_train.npy' --alpha 10 --beta 25 --gamma 1 --lr 0.01 --fit_epochs 1200
screen -d -m python paper_style_transfer.py 'data/energy/preprocessed/in_sample_test.npy' 'data/energy/preprocessed/style_test.npy' --alpha 10 --beta 25 --gamma 1 --lr 0.01 --fit_epochs 1200

## Perturbed Dataset.
screen -d -m python paper_style_transfer.py 'data/energy/preprocessed/perturbed_train.npy' 'data/energy/preprocessed/style_train.npy' --alpha 10 --beta 25 --gamma 1 --lr 0.01 --fit_epochs 1200
screen -d -m python paper_style_transfer.py 'data/energy/preprocessed/perturbed_test.npy' 'data/energy/preprocessed/style_test.npy' --alpha 10 --beta 25 --gamma 1 --lr 0.01 --fit_epochs 1200

# GOOGLE STOCKS
## IN SAMPLE
screen -d -m python paper_style_transfer.py 'data/google_stocks/preprocessed/in_sample_train.npy' 'data/google_stocks/preprocessed/style_train.npy' --alpha 10 --beta 25 --gamma 1 --lr 0.01 --fit_epochs 1200
screen -d -m python paper_style_transfer.py 'data/google_stocks/preprocessed/in_sample_test.npy' 'data/google_stocks/preprocessed/style_test.npy' --alpha 10 --beta 25 --gamma 1 --lr 0.01 --fit_epochs 1200

## PERTURBED
screen -d -m python paper_style_transfer.py 'data/google_stocks/preprocessed/perturbed_train.npy' 'data/google_stocks/preprocessed/style_train.npy' --alpha 10 --beta 25 --gamma 1 --lr 0.01 --fit_epochs 1200
screen -d -m python paper_style_transfer.py 'data/google_stocks/preprocessed/perturbed_test.npy' 'data/google_stocks/preprocessed/style_test.npy' --alpha 10 --beta 25 --gamma 1 --lr 0.01 --fit_epochs 1200