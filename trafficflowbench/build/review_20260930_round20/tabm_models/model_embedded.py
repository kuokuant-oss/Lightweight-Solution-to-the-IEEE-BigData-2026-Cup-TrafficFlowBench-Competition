"""Author TabM with author numerical embeddings; candidate resource probe only."""
from model import weighted_member_bce, probabilities, torch, tabm
from rtdl_num_embeddings import LinearReLUEmbeddings

CONFIG = dict(d_out=1, n_blocks=2, d_block=128, dropout=0.1, k=32,
              arch_type='tabm', numerical_embedding='LinearReLUEmbeddings',
              embedding_dim=4)


def make_model(features):
    return tabm.TabM.make(n_num_features=features, d_out=1,
                         n_blocks=2, d_block=128, dropout=0.1, k=32,
                         arch_type='tabm',
                         num_embeddings=LinearReLUEmbeddings(features, 4))
