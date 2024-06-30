import os
import csv
import numpy as np
from pathlib import Path
from fedot.api.main import Fedot
from fedot.core.data.multi_modal import MultiModalData
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

targets = {'product_sentiment_machine_hack': ('Sentiment', 'classification', 'acc'),
           'data_scientist_salary': ('salary', 'classification', 'acc'),
           'melbourne_airbnb': ('price_label', 'classification', 'acc'),
           'news_channel': ('channel', 'classification', 'acc'),
           'wine_reviews': ('variety', 'classification', 'acc'),
           'imdb_genre_prediction': ('Genre_is_Drama', 'classification', 'roc_auc'),
           'fake_job_postings2': ('fraudulent', 'classification', 'roc_auc'),
           'kick_starter_funding': ('final_status', 'classification', 'roc_auc'),
           'jigsaw_unintended_bias100K': ('target', 'classification', 'roc_auc'),
           'google_qa_answer_type_reason_explanation': ('answer_type_reason_explanation', 'regression', 'r2'),
           'google_qa_question_type_reason_explanation': ('question_type_reason_explanation', 'regression', 'r2'),
           'bookprice_prediction': ('Price', 'regression', 'r2'),
           'jc_penney_products': ('sale_price', 'regression', 'r2'),
           'women_clothing_review': ('Rating', 'regression', 'r2'),
           'ae_price_prediction': ('price', 'regression', 'r2'),
           'news_popularity2': ('log_shares', 'regression', 'r2'),
           'california_house_price': ('Sold Price', 'regression', 'r2'),
           'mercari_price_suggestion100K': ('price', 'regression', 'r2')}


def get_text_sources_names(data: MultiModalData) -> list:
    text_sources = [source.split('/')[1] for source in list(data.keys()) if 'data_source_text' in source]
    return text_sources


def save_to_csv(results: list):
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for sublist in results:
            writer.writerow(sublist)


def run_multimodal_dataset(dataset_name: str, timeout: int = 1, n_jobs: int = 4):
    print(f'Fit of dataset {dataset_name} is started')
    try:
        target, task, metric_name = targets.get(dataset_name)
        fit_path = Path('data/', f'{dataset_name}_train.csv')
        predict_path = Path('data/', f'{dataset_name}_test.csv')

        fit_data = MultiModalData.from_csv(file_path=fit_path, task=task, target_columns=target, index_col=None)

        predict_data = MultiModalData.from_csv(file_path=predict_path, task=task, target_columns=target, index_col=None,
                                               text_columns=get_text_sources_names(fit_data))

        automl_model = Fedot(problem=task, timeout=timeout, n_jobs=n_jobs, safe_mode=False, metric=metric_name)
        automl_model.fit(features=fit_data,
                         target=fit_data.target)

        prediction = automl_model.predict(predict_data)

        if metric_name == 'acc':
            metric = accuracy_score(predict_data.target, prediction)
        elif metric_name == 'roc_auc':
            metric = roc_auc_score(predict_data.target, prediction)
        elif metric_name == 'r2':
            metric = r2_score(predict_data.target, prediction)

        print(f'dataset {dataset_name} successfully finished with {metric_name} = {np.round(metric, 3)}')
        automl_model.history.save(f'history_{dataset_name}.json')
        automl_model.current_pipeline.save(f'pipeline_{dataset_name}')
        return metric_name, np.round(metric, 3)
    except Exception as ex:
        print(f'dataset {dataset_name} failed')
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return 'error', 'error'


if __name__ == '__main__':
    results = []
    for dataset in os.listdir('data'):
       if '_train' in dataset:
           dataset_name = str(dataset.split('_train')[0])
           metric_name, metric = run_multimodal_dataset(dataset_name=dataset_name, timeout=1)
           results.append([dataset_name, metric_name, metric])
    save_to_csv(results)
