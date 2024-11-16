import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# this is a general function for getting scores from a model dataset
# Don't care about it
def get_models_scores(model_dataset,
               model_score_function,
               progress,
               live=True,
               strict=False):
    '''
    This is a general function for getting scores from a model dataset.
    model_dataset: an iterable that contains models
    model_score_function: a function that takes a model as input and returns a score
    progress: if True, it will print the progress of the function
    '''
    labels = []
    scores = []

    tq = range(len(model_dataset))
    if progress is False:
        tq = tqdm(tq)
    
    if live:
        seen_labels = set()
    failed_models = 0
    
    for i in tq:
        try:
            model, label = model_dataset[i]

            score = model_score_function(model)
            if progress:
                print(f'No. {i}, Label: {label}, Score: {score}')
            
            scores.append(score)
            labels.append(label)
            if live:
                print("Label:", label, "Score:", score)
                seen_labels.add(label)
                
                if len(seen_labels) > 1:
                    print("Current auc:", roc_auc_score(labels, scores))
        except Exception as e:
            if strict:
                raise e
            failed_models += 1
            print(f"The following error occured during the evaluation of a model with name {model.meta_data.get('name') if model else 'NaN'}: {str(e)}")
            print("Skipping this model")
    print("No. of failed models:", failed_models)
    return scores, labels

# All it does is that it iterates over the model dataset, calculate some score bease on the model and the dataloader
# and finally returns auc on the scores
def get_auc_on_models_scores(model_dataset,
                           score_function,
                           dataloader,
                           other_score_function_params={},
                           dataloader_func=None,
                           progress=False):
    '''
    This function calculates the AUC of the model on the given model dataset.
    model_dataset: an iterable that contains models
    score_function: a function that takes a model as input and returns a score
    progress: if True, it will print the progress of the function
    '''
    assert dataloader is not None or dataloader_func is not None

    # this is a function that just calls the score function on the model
    # the purpose of this function is to be compatible with get_models_scores
    
    def model_score_function(model):
        final_dataloader = dataloader_func(model) if dataloader_func is not None else dataloader
        
        return score_function(model, final_dataloader, progress=progress, **other_score_function_params)
    
    scores, labels = get_models_scores(model_dataset, model_score_function, progress)
    
    return roc_auc_score(labels, scores)
