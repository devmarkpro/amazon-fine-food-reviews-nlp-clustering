from fine_food_review import FineFoodReview
from word_tokenizer import WordTokenizer

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():

    # tokenizer = WordTokenizer()
    # text = """<head>This is a sample sentence for tokenization. <br/>        !it includes @# special characters.

    # it also has some numbers like 123 and punctuation marks like commas, periods, and exclamation points.

    # it also has new lines and extra spaces. 😁 >hr> <br>

    # some non english sentences like "C'est la vie" or "¡Hola, mundo!" are also included.
    # """
    # tokens = tokenizer(text, return_tokens=False)
    # print("-----------------------------------------------------")
    # print("Tokenized Text:")
    # print(tokens)
    FFR = FineFoodReview(
        dataset_file_name="fine_food_reviews_tokenized.csv", dataset_path="./data"
    )
    FFR.tokenize_reviews()
    # doc_vector_tfidf = FFR.vectorize_reviews(
    #     method="tfidf", column="ReviewVector_tfidf", force=True
    # )

    doc_vector_tfidf = FFR.vectorize_reviews(
        method="glove", column="ReviewVector_glove", force=True
    )

    logger.info("stored vectorized reviews into the disk")
    # store new dataset in the disk
    FFR.dataset.to_csv("data/fine_food_reviews_vectorized.csv", index=False)
    logger.info("Vectorized DataFrame:")
    logger.info(FFR.dataset.head())


if __name__ == "__main__":
    main()
