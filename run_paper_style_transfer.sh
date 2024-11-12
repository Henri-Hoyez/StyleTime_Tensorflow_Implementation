#/bin/sh



style_train="data/energy/preprocessed/style_train.npy"
style_test="data/energy/preprocessed/style_test.npy"


python paper_style_transfer.py "data/energy/preprocessed/in_sample_train.npy" $style_train
python paper_style_transfer.py "data/energy/preprocessed/in_sample_test.npy" $style_test

# Perturbed Dataset.
python paper_style_transfer.py "data/energy/preprocessed/perturbed_train.npy" $style_train
python paper_style_transfer.py "data/energy/preprocessed/perturbed_test.npy" $style_test

