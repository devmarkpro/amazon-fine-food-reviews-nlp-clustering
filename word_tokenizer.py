import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from typing import List, Union
import html
import logging

logger = logging.getLogger(__name__)


# Download necessary resources only once
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class WordTokenizer:
    """
    A class for tokenizing and preprocessing text data using NLTK.
    This class handles text normalization, punctuation removal, and optional stop word filtering.

    Attributes:
        remove_stopwords (bool): Whether to remove stop words.
        lower_case (bool): Whether to convert text to lowercase.
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lower_case: bool = True,
        use_lemmatization: bool = True,
        replace_repeated_chars: bool = True,
    ):
        self.remove_stopwords = remove_stopwords
        self.lower_case = lower_case
        self.lematizer = nltk.WordNetLemmatizer() if use_lemmatization else None
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        self.replace_repeated_chars = replace_repeated_chars

    def process(
        self, text: Union[str, List[str]], return_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Process the input text by cleaning, normalizing, and optionally removing stop words.

        Args:
            text (str | List[str]): The input text. If a list is provided, its elements will be joined into a single string.
            return_tokens (bool): If True, returns a list of tokens instead of a string.

        Returns:
            str | List[str]: The processed text as a string or list of tokens.
        """
        if not text:
            return [] if return_tokens else ""
        logger.debug(f"Processing text: {text}")

        text_str = " ".join(map(str, text)) if isinstance(text, list) else str(text)

        text = self._clean_text(text_str)
        logger.debug(f"Cleaned text: {text}")

        tokens = self._tokenize(text)
        logger.debug(f"Tokenized text: {tokens}")

        tokens = self._remove_stopwords(tokens)
        logger.debug(f"Tokens after stop word removal: {tokens}")

        tokens = self._apply_lemmatization(tokens)
        logger.debug(f"Tokens after lemmatization: {tokens}")

        tokens = [token for token in tokens if token]  # Remove empty tokens

        logger.debug(f"Final tokens: {tokens}")

        return tokens if return_tokens else " ".join(tokens)

    def _clean_text(self, text: str) -> str:
        if self.lower_case:
            text = text.lower()

        text = html.unescape(text)  # decode &amp; and other HTML entities
        text = text.strip().replace("\n", " ").replace("\r", " ")
        text = re.sub(r"<[^>]*>", "", text)  # Remove HTML tags
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
        text = self._fix_contractions(text)  # Fix common contractions

        # reduce repeated characters like "sooo good" to "so good" or "loooove" to "love"
        if self.replace_repeated_chars:
            text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        return text

    def _apply_lemmatization(self, tokens: List[str]) -> List[str]:
        if self.lematizer is not None:
            tokens = [self.lematizer.lemmatize(token) for token in tokens]
        return tokens

    def _tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        if self.remove_stopwords:
            tokens = self._remove_stop_words(tokens)
        return tokens

    def _remove_stop_words(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.stop_words]

    def _fix_contractions(self, text: str) -> str:
        contraction_map = {
            "dont": "do not",
            "cant": "can not",
            "wont": "will not",
            "isnt": "is not",
            "arent": "are not",
            "wasnt": "was not",
            "werent": "were not",
            "didnt": "did not",
            "doesnt": "does not",
            "havent": "have not",
            "hasnt": "has not",
            "hadnt": "had not",
            "shouldnt": "should not",
            "wouldnt": "would not",
            "couldnt": "could not",
            "mustnt": "must not",
            "mightnt": "might not",
            "neednt": "need not",
            "im": "i am",
            "ive": "i have",
            "ill": "i will",
            "id": "i would",
            "youre": "you are",
            "youve": "you have",
            "youll": "you will",
            "youd": "you would",
            "theyre": "they are",
            "theyve": "they have",
            "theyll": "they will",
            "theyd": "they would",
            "weve": "we have",
            "theres": "there is",
            "heres": "here is",
            "thats": "that is",
            "whats": "what is",
            "whos": "who is",
            "don't": "do not",
            "can't": "can not",
            "won't": "will not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "didn't": "did not",
            "doesn't": "does not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "mustn't": "must not",
            "mightn't": "might not",
            "needn't": "need not",
            "i'm": "i am",
            "i've": "i have",
            "i'll": "i will",
            "i'd": "i would",
            "you're": "you are",
            "you've": "you have",
            "you'll": "you will",
            "you'd": "you would",
            "they're": "they are",
            "they've": "they have",
            "they'll": "they will",
            "they'd": "they would",
            "we've": "we have",
        }
        for contraction, full_form in contraction_map.items():
            text = re.sub(r"\b" + re.escape(contraction) + r"\b", full_form, text)
        return text

    def __call__(
        self, text: Union[str, List[str]], return_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Allow the object to be called like a function.

        Example:
            tokenizer = WordTokenizer()
            clean_text = tokenizer("Some input text here.")
        """
        return self.process(text, return_tokens=return_tokens)
