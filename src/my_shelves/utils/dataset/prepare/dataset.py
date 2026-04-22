"""
Module to retrieve data from csv
"""

if __name__ == "__main__":
    import read
    import my_shelves.utils.dataset.prepare.clean as clean
    import my_shelves.utils.dataset.prepare.utils as utils
    DATA_DIR = "../../data"
    books_df = read.get_books(DATA_DIR, lang='ENG', nrows=1000)
    reviews_df = read.get_reviews(DATA_DIR, lang='ENG', nrows=1000)
    print(books_df.head())
    print(reviews_df.head())
    # Clean the reviews text by removing rows with missing or empty review text
    reviews_df = clean.clean_text(reviews_df, column_name="review_text")
    # Add a new column that counts the number of words in the review text
    reviews_df = utils.add_reviews_count_words(reviews_df, column_name="review_text")
    # Clean the reviews by removing rows where the number of words in the review
    #  text is less than or equal to 5
    reviews_df = clean.clean_small_text(reviews_df,
                                         column_name="review_text",
                                         min_words=5)
    # Clean the date columns in the reviews DataFrame by converting them to
    # datetime format and replacing invalid entries with appropriate default dates
    reviews_df = clean.clean_dates(reviews_df)
    # Compute the read duration in hours by calculating the difference between
    # the "read_at" and "started_at" datetime columns and adding a new column
    # to the DataFrame with the result
    reviews_df = utils.compute_read_duration(reviews_df,
                                       start_column="started_at",
                                       end_column="read_at")
    print(reviews_df.head())
