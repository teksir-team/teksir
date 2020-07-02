import pandas as pd

import transformers


class Augmentator:
    """
    Base Augmentator class.
    """

    def __init__(self, model_name: str) -> None:

        self.model_name = model_name
    
    def augment(self):
        raise NotImplementedError


class BertAugmentator(Augmentator):
    """
    Augmentator class which uses BERT as a contextual model to augment the input data.
    """
    
    def __init__(self, model_name: str) -> None:
        super(BertAugmentator, self).__init__(model_name=model_name)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.BertForMaskedLM.from_pretrained(model_name)
        self.pipeline = transformers.pipeline("fill-mask", model=model, tokenizer=tokenizer)
        

    def augment(self, df_train: pd.DataFrame):
        """
        @param df_train: DataFrame object which holds the training data as (X, y) pairs.
        """
        # TODO: How to use this function?
        pass