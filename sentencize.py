import re
from spacy.lang.tr import Turkish
from typing import Union, List

DIGIT_BOUNDARY = re.compile('^[0-9]+$')


class Sentencizer:

    def __init__(self):
        self.nlp = Turkish()
        self.nlp.add_pipe(prevent_digit_seg)
        self.nlp.add_pipe(prevent_quote_and_bracket_seg, name='prevent-sbd')
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def sentencize(self, text: Union[str, list]) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        docs = list(self.nlp.pipe(text))
        sents = [[sent.text for sent in doc.sents] for doc in docs]

        return sents


def prevent_digit_seg(doc):
    prev = doc[0].text
    length = len(doc)

    for index, token in enumerate(doc):

        if (token.text == '.' and DIGIT_BOUNDARY.match(prev) and index != (length - 1)):
            doc[index + 1].is_sent_start = False

        prev = token.text

    return doc


def prevent_quote_and_bracket_seg(doc):
    """ Ensure that SBD does not run on tokens inside quotation marks and brackets. """
    quote_open = False
    bracket_open = False
    can_sbd = True

    for index, token in enumerate(doc):
        # Don't do sbd on these tokens
        if not can_sbd:
            token.is_sent_start = False

        # Not using .is_quote so that we don't mix-and-match different kinds of quotes (e.g. ' and ")
        # Especially useful since quotes don't seem to work well with .is_left_punct or .is_right_punct
        if token.text == '"':
            quote_open = False if quote_open else True

        elif token.is_bracket and token.is_left_punct:
            bracket_open = True

        elif token.is_bracket and token.is_right_punct:
            bracket_open = False

        can_sbd = not (quote_open or bracket_open)

    return doc
