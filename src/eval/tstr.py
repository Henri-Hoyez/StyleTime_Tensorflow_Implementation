from src.models.naive_discriminator import make_naive_discriminator

def train_naive_discriminator(train_dset, valid_dset, seq_shape, epochs, n_classes):

    naive_discr = make_naive_discriminator(seq_shape, n_classes)
    history = naive_discr.fit(train_dset, validation_data=valid_dset, epochs=epochs, verbose=1)

    return naive_discr.evaluate(valid_dset)[1], history