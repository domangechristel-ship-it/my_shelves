import os
import pandas as pd

from prefect import task, flow
# from prefect.artifacts import create_table_artifact


from my_shelves.ml.classification.classifiers import (SentimentClassifier,
                                                      EmotionClassifier,
                                                      ZeroShotClassifier)

from my_shelves.ml.classification.params import (PREFECT_FLOW_NAME,
                                                 DATASET_ROOT,
                                                 OUTPUT_FOLDER)


@task
def train_emotions(X: pd.DataFrame, n_rows: str = "10k") -> str:
    base_name = 'emotion_from_description'
    output_csv = f'{OUTPUT_FOLDER}/{base_name}_{n_rows}.csv'

    # if os.path.exists(output_csv):
    #     print(f"{base_name}_{n_rows}.csv exists, skipping training...")
    #     return output_csv

    emotion = EmotionClassifier(
        batch_size=32,
        device=0  # GPU si dispo
    )

    df = emotion.predict_and_save(X["description"],
                                  X["book_id"],
                                  filename=output_csv)
    return output_csv


@task
def train_content_intensity(X: pd.DataFrame, n_rows: str = "10k") -> str:
    base_name = 'content_intensity_from_description'
    output_csv = f'{OUTPUT_FOLDER}/{base_name}_{n_rows}.csv'

    # if os.path.exists(output_csv):
    #     print(f"{base_name}_{n_rows}.csv exists, skipping training...")
    #     return output_csv

    content_intensity = ZeroShotClassifier(
        task='content_intensity',
        batch_size=32,
        device=0  # GPU si dispo
    )

    df = content_intensity.predict_and_save(X["description"],
                                            X["book_id"],
                                            filename=output_csv)
    return output_csv


@task
def train_romance_heat_level(X: pd.DataFrame, n_rows: str = "10k"):
    base_name = 'romance_heat_level_from_description'
    output_csv = f'{OUTPUT_FOLDER}/{base_name}_{n_rows}.csv'

    # if os.path.exists(output_csv):
    #     print(f"{base_name}_{n_rows}.csv exists, skipping training...")
    #     return output_csv

    romance_heat_level = ZeroShotClassifier(
        task='romance_heat_level',
        batch_size=32,
        device=0  # GPU si dispo
    )

    df = romance_heat_level.predict_and_save(X["description"],
                                             X["book_id"],
                                             filename=output_csv)
    return output_csv


@task
def train_character_type(X: pd.DataFrame, n_rows: str = "10k"):
    base_name = 'character_type_from_description'
    output_csv = f'{OUTPUT_FOLDER}/{base_name}_{n_rows}.csv'

    # if os.path.exists(output_csv):
    #     print(f"{base_name}_{n_rows}.csv exists, skipping training...")
    #     return output_csv

    character_type= ZeroShotClassifier(
        task='character_type',
        batch_size=32,
        device=0  # GPU si dispo
    )

    df = character_type.predict_and_save(X["description"],
                                         X["book_id"],
                                         filename=output_csv)
    return output_csv


@task
def train_main_themes(X: pd.DataFrame, n_rows: str = "10k"):
    base_name = 'main_themes_from_description'
    output_csv = f'{OUTPUT_FOLDER}/{base_name}_{n_rows}.csv'

    # if os.path.exists(output_csv):
    #     print(f"{base_name}_{n_rows}.csv exists, skipping training...")
    #     return output_csv

    main_themes= ZeroShotClassifier(
        task='main_themes',
        batch_size=32,
        device=0  # GPU si dispo
    )

    df = main_themes.predict_and_save(X["description"],
                                      X["book_id"],
                                      filename=output_csv)
    return output_csv


@task
def train_pace(X: pd.DataFrame, n_rows: str = "10k"):
    base_name = 'pace_from_reviews'
    output_csv = f'{OUTPUT_FOLDER}/{base_name}_{n_rows}.csv'

    # if os.path.exists(output_csv):
    #     print(f"{base_name}_{n_rows}.csv exists, skipping training...")
    #     return output_csv

    pace = ZeroShotClassifier(
        task='pace',
        batch_size=32,
        device=0  # GPU si dispo
    )

    df = pace.predict_and_save(X["review_text"],
                               X["book_id"],
                               filename=output_csv)
    return output_csv


@task
def train_sentiment(X: pd.DataFrame, n_rows: str = "10k"):
    base_name = 'sentiment_from_reviews'
    output_csv = f'{OUTPUT_FOLDER}/{base_name}_{n_rows}.csv'

    # if os.path.exists(output_csv):
    #     print(f"{base_name}_{n_rows}.csv exists, skipping training...")
    #     return output_csv

    sentiment = SentimentClassifier(
        batch_size=32,
        device=0  # GPU si dispo
    )

    df = sentiment.predict_and_save(X["review_text"],
                                    X["book_id"],
                                    filename=output_csv)
    return output_csv

@task
def train_merge_results (n_rows: str = "10k"):

    base_name = 'merged_features'
    output_csv = f'{OUTPUT_FOLDER}/{base_name}_{n_rows}.csv'

    if os.path.exists(output_csv):
        print(f"{base_name}_{n_rows}.csv exists, skipping merging...")
        return output_csv


    emotions = pd.read_csv(f'{OUTPUT_FOLDER}/emotion_from_description_{n_rows}.csv')
    content_intensity = pd.read_csv(f'{OUTPUT_FOLDER}/content_intensity_from_description_{n_rows}.csv')
    romance_heat_level = pd.read_csv(f'{OUTPUT_FOLDER}/romance_heat_level_from_description_{n_rows}.csv')
    character_type = pd.read_csv(f'{OUTPUT_FOLDER}/character_type_from_description_{n_rows}.csv')
    main_themes = pd.read_csv(f'{OUTPUT_FOLDER}/main_themes_from_description_{n_rows}.csv')
    pace = pd.read_csv(f'{OUTPUT_FOLDER}/pace_from_reviews_{n_rows}.csv')
    sentiment = pd.read_csv(f'{OUTPUT_FOLDER}/sentiment_from_reviews_{n_rows}.csv')

    df = (emotions[['book_id','emotions']]
        .merge(content_intensity[['book_id','content_intensity']], on='book_id')
        .merge(romance_heat_level[['book_id','romance_heat_level']], on='book_id')
        .merge(character_type[['book_id','character_type']], on='book_id')
        .merge(main_themes[['book_id','main_themes']], on='book_id')
        .merge(pace[['book_id','pace']], on='book_id')
        .merge(sentiment[['book_id','sentiment']], on='book_id')
    )

    df.to_csv(output_csv)

    return output_csv


@flow(name=PREFECT_FLOW_NAME, retries=3)
def train_flow(n_rows: str = "10k"):
    df = pd.read_csv(f"{DATASET_ROOT}/base_ENG_{n_rows}.csv",
                     # nrows=10000,
                     usecols=["book_id", "description", "review_text"])
    X = df.copy()

    X.dropna(inplace=True)

    train_emotions_task = train_emotions.submit(X, n_rows).wait()

    train_content_intensity_task = train_content_intensity.submit(X, n_rows).wait()

    train_romance_heat_level_task = train_romance_heat_level.submit(X, n_rows).wait()

    train_character_type_task = train_character_type.submit(X, n_rows).wait()

    train_main_themes_task = train_main_themes.submit(X, n_rows).wait()

    train_pace_task = train_pace.submit(X, n_rows).wait()

    train_sentiment_task = train_sentiment.submit(X, n_rows).wait()

    train_merge_results_task = train_merge_results.submit(
        n_rows,
        wait_for=[
            train_emotions_task,
            train_content_intensity_task,
            train_romance_heat_level_task,
            train_character_type_task,
            train_main_themes_task,
            train_pace_task,
            train_sentiment_task,
        ]
    ).wait()



if __name__ == "__main__":
    n_rows = "20k"
    train_flow(n_rows=n_rows)
