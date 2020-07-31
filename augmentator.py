import sys
import random

import pandas as pd

import transformers


class Augmentator:
    """
    Base Augmentator class.
    """

    def __init__(self, model_name: str, augmentation_config: dict) -> None:

        self.augmentation_config = augmentation_config
    
    def augment(self):
        raise NotImplementedError


class BertAugmentator(Augmentator):
    """
    Augmentator class which uses BERT as a contextual model to augment the input data.
    """
    
    def __init__(self, augmentation_config: dict) -> None:
        super(BertAugmentator, self).__init__(augmentation_config=augmentation_config)
        
        model_name = augmentation_config.get("model_name", "dbmdz/bert-base-turkish-cased")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.BertForMaskedLM.from_pretrained(model_name)
        self.pipeline = transformers.pipeline("fill-mask", model=model, tokenizer=tokenizer)
        

    def augment(self, sentence: str) -> str:
        """
        """
        return self._augment_sentence(sentence)
        

    def _augment_sentence(self, sentence: str) -> str:
        """
        Augment the sentence by masking random tokens and making predictions using BERT.
        """
        
        tokenized_sent = " ".join(self.pipeline.tokenizer.tokenize(sentence)).replace(" ##", "")
        sent_tokens = tokenized_sent.split()
        len_sent_tokens = len(sent_tokens)
        if len_sent_tokens < 2:
            return sentence

        replace_token_count = int(len_sent_tokens * self.augmentation_config["frac"])
        # We are taking the first 5 index to make the generation faster.
        random_token_indices = random.sample(range(len_sent_tokens), replace_token_count)[:5]
        for rand_idx in random_token_indices:
            sent_tokens[rand_idx] = self.pipeline.tokenizer.mask_token
            masked_sent = " ".join(sent_tokens)
            predictions = self.pipeline(masked_sent)
            replaced_token = self.pipeline.tokenizer.ids_to_tokens[predictions[0]["token"]]
            sent_tokens[rand_idx] = replaced_token
        
        return " ".join(sent_tokens)