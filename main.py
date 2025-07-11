from fine_food_review import FineFoodReview
from word_tokenizer import WordTokenizer

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    tokenizer = WordTokenizer()
    text = """<head>This is a sample sentence for tokenization. <br/>        !it includes @# special characters.

    it also has some numbers like 123 and punctuation marks like commas, periods, and exclamation points.

    it also has new lines and extra spaces. 😁 >hr> <br>

    some non english sentences like "C'est la vie" or "¡Hola, mundo!" are also included.
    """
    tokens = tokenizer(text, return_tokens=False)
    print("-----------------------------------------------------")
    print("Tokenized Text:")
    print(tokens)


if __name__ == "__main__":
    main()
