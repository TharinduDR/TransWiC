from torch.utils.data import Dataset
from typing import List, Union
from .. import siamese_transwic


class EncodeDataset(Dataset):
    def __init__(self,
                 sentences: Union[List[str], List[int]],
                 model: siamese_transwic,
                 is_tokenized: bool = True):
        """
        EncodeDataset is used by SentenceTransformer.encode method. It just stores
        the input texts and returns a tokenized version of it.
        """
        self.model = model
        self.sentences = sentences
        self.is_tokenized = is_tokenized

    def __getitem__(self, item):
        return self.sentences[item] if self.is_tokenized else self.model.tokenize(self.sentences[item])

    def __len__(self):
        return len(self.sentences)
