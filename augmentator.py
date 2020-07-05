import sys
import random

import pandas as pd

import transformers

from sentencize import Sentencizer


class Augmentator:
    """
    Base Augmentator class.
    """

    def __init__(self, model_name: str, augmentation_config: dict) -> None:

        self.model_name = model_name
        self.augmentation_config = augmentation_config
        self.sentencizer = Sentencizer()
    
    def augment(self):
        raise NotImplementedError


class BertAugmentator(Augmentator):
    """
    Augmentator class which uses BERT as a contextual model to augment the input data.
    """
    
    def __init__(self, model_name: str, augmentation_config: dict) -> None:
        super(BertAugmentator, self).__init__(model_name=model_name, augmentation_config=augmentation_config)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.BertForMaskedLM.from_pretrained(model_name)
        self.pipeline = transformers.pipeline("fill-mask", model=model, tokenizer=tokenizer)
        

    def augment(self, df_train: pd.DataFrame):
        """
        :param df_train: DataFrame object which holds the training data as (X, y) pairs.
        """        
        augmentation_data = df_train.sample(frac=self.augmentation_config["frac"])
        self.augmentation_indices = augmentation_data.index

        print("Sentencizing the data", file=sys.stderr)
        augmentation_data["text"] = augmentation_data["text"].apply(self.sentencizer.sentencize)
        augmented_samples, augmented_labels = [], []
        # TODO: Doing the replacement on the changed version of the sentence
        # vs. initial version.

        for idx, sample in augmentation_data.iterrows():
            label = sample.label
            sents = sample.text[0]
            augmented_sentences = []
            for sent in sents:    
                augmented_sentences.append(self._augment_sentence(sent))

            augmented_samples.append(" ".join(augmented_sentences))
            augmented_labels.append(label)

        return augmented_samples, augmented_labels

    def _augment_sentence(self, sentence: str) -> str:
        """
        Augment the sentence by masking random tokens and making predictions using BERT.
        """
        
        tokenized_sent = " ".join(self.pipeline.tokenizer.tokenize(sentence)).replace(" ##", "")
        sent_tokens = tokenized_sent.split()
        if len(sent_tokens) < 2:
            return sentence
        # Randomly change 5 tokens without any condition.
        # TODO:
        # This value will be a parameter. Also we might put some conditions here for punctuations and digits.
        for i in range(5):
            rand_idx = random.randrange(0, len(sent_tokens) - 1)
            sent_tokens[rand_idx] = self.pipeline.tokenizer.mask_token
            masked_sent = " ".join(sent_tokens)
            predictions = self.pipeline(masked_sent)
            replaced_token = self.pipeline.tokenizer.ids_to_tokens[predictions[0]["token"]]
            sent_tokens[rand_idx] = replaced_token
        
        return " ".join(sent_tokens)