import torch
from torchkge.sampling import NegativeSampler


class CrossNegativeSampler(NegativeSampler):
    """Description.

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, kg, kg_val=None, kg_test=None, n_neg=1):
        super().__init__(kg, kg_val, kg_test, n_neg)

        self.unique_tails = kg.tail_idx.unique()

    def corrupt_batch(self, heads, tails, relations=None, n_neg=None):
        """For each true triplet, produce a corrupted one not different from
        any other true triplet. If `heads` and `tails` are cuda objects ,
        then the returned tensors are on the GPU.
        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of heads of the relations in the
            current batch.
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of tails of the relations in the
            current batch.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of relations in the current
            batch. This is optional here and mainly present because of the
            interface with other NegativeSampler objects.
        n_neg: int (opt)
            Number of negative sample to create from each fact. It overwrites
            the value set at the construction of the sampler.
        Returns
        -------
        neg_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled heads of
            the relations in the current batch.
        neg_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled tails of
            the relations in the current batch.
        """
        if n_neg is None:
            n_neg = self.n_neg

        device = heads.device
        assert (device == tails.device)

        batch_size = heads.shape[0]
        neg_heads = heads.repeat(n_neg)
        neg_tails = tails.repeat(n_neg)

        # здесь мы понимам, какое число надо выкинуть из рассмотрения
        neg_tails_help = neg_tails.view(-1, 1) - self.unique_tails

        # здесь в каждой строке будут варианты для случайного выбора неправильного концепта
        neg_tails_variants = neg_tails.view(-1, 1) - neg_tails_help[neg_tails_help != 0].view(
            neg_tails.size(0), self.unique_tails.size(0)-1)

        neg_tails = neg_tails_variants[torch.arange(neg_tails_variants.size(0)),
                                       torch.randint(0, neg_tails_variants.size(1), size=(neg_tails_variants.size(0),))]

        return neg_heads.long(), neg_tails.long()
