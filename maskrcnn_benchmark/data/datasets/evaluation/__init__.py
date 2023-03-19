from maskrcnn_benchmark.data import datasets

# from .word import word_evaluation
from .detection.ic15 import ic15_detection_evaluation
from .retrieval.svt import svt_retrieval_evaluation
from .retrieval.iiit import iiit_retrieval_evaluation
from .retrieval.cocotext import cocotext_retrieval_evaluation
from .retrieval.ctw import ctw_retrieval_evaluation
from .retrieval.chinese_collect import chinese_collect_retrieval_evaluation
from .retrieval.totaltext import totaltext_retrieval_evaluation
def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    # if isinstance(dataset, datasets.WordDataset):
    #     return word_evaluation(**args)

    if isinstance(dataset, datasets.SVTDataset):
        return svt_retrieval_evaluation(**args)
    elif isinstance(dataset, datasets.IIITDataset):
        return iiit_retrieval_evaluation(**args)
    elif isinstance(dataset, datasets.Icdar15Dateset):
        return iiit_retrieval_evaluation(**args)
    elif isinstance(dataset, datasets.ArTDataset):
        return iiit_retrieval_evaluation(**args)
    elif isinstance(dataset, datasets.COCOTextDataset):
        return cocotext_retrieval_evaluation(**args)
    elif isinstance(dataset, datasets.ChineseCollectDataset):
        return chinese_collect_retrieval_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
