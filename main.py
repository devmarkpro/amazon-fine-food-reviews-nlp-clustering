from fine_food_review import FineFoodReview


def main():
    amazon_reviews = FineFoodReview(dataset_path="./data")
    summary = amazon_reviews.eda.get_summary()
    print(summary)


if __name__ == "__main__":
    main()
