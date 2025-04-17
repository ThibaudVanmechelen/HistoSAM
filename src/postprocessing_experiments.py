import os
import gc
import torch

from torch.utils.data import DataLoader
from model.sam2_model import TrainableSAM2
from model.model import load_model
from dataset_processing.dataset import SamDatasetFromFiles, filter_dataset
from dataset_processing.preprocess import collate_fn
from .evaluate import test_loop
from utils.config import load_config
from utils.save_scores import save_scores


def run_postprocessing_testing(dataset_path : str, 
                               config_path : str, 
                               checkpoint_path : str, 
                               is_sam2 : bool, 
                               output_dir_path : str,
                               testing_name : str, 
                               use_dataset : list[bool], 
                               last_model_path : str = None,
                               post_process_type : str = 'standard'):
    """
    Function to run the experiment on the different post processing types.

    Args:
        dataset_path (str): path to the dataset.
        config_path (str): path to the config.
        checkpoint_path (str): path to the checkpoint.
        is_sam2 (bool): whether we test sam or sam2.
        output_dir_path (str): path where to save the testing file.
        testing_name (str): name of the testing file.
        use_dataset (list[bool]): which dataset to use in dataset_path.
        last_model_path (str, optional): path to the last model if some weights must be loaded. Defaults to None.
        post_process_type (str, optional): which postprocessing to apply. Defaults to 'standard'.
    """
    print("Loading the configs")
    config = load_config(config_path)

    print("Creating the dataset...")
    prompt_type = {
        'points' : config.testing.points,
        'box' : config.testing.box,
        'neg_points' : config.testing.negative_points,
        'mask' : config.testing.mask_prompt
    }

    dataset = SamDatasetFromFiles(
        root = dataset_path,
        transform = None,
        use_img_embeddings = False,
        prompt_type = prompt_type,
        n_points = config.testing.n_points,
        n_neg_points = config.testing.n_neg_points,
        verbose = True,
        to_dict = True,
        is_sam2_prompt = is_sam2,
        neg_points_inside_box = config.testing.negative_points_inside_box,
        points_near_center = config.testing.points_near_center,
        random_box_shift = config.testing.random_box_shift,
        mask_prompt_type = config.testing.mask_prompt_type,
        box_around_mask = config.testing.box_around_prompt_mask,
        filter_files = lambda x: filter_dataset(x, use_dataset),
        load_on_cpu = True
    )

    dataloader = DataLoader(dataset, batch_size = config.testing.batch_size, shuffle = False, collate_fn = collate_fn)

    if not is_sam2:
        print("Starting Testing for SAM...")
        model = load_model(checkpoint_path, config.sam.model_type, img_embeddings_as_input = False, return_iou = True).to(config.misc.device)

        if last_model_path:
            print("Loading the last model...")
            model.load_state_dict(torch.load(last_model_path))

        print('# Starting Without Post-Processing... #')
        scores_no_postpro = test_loop(model, dataloader, config.misc.device, config.sam.input_mask_eval, return_mean = False, is_eval_post_processing = True, do_post_process = False)

        print('# Starting With Post-Processing... #')
        scores_postpro = test_loop(model, dataloader, config.misc.device, config.sam.input_mask_eval, return_mean = False, is_eval_post_processing = True, do_post_process = True, post_process_type = post_process_type)

    else:
        print("Starting Testing for SAM2...")
        model = TrainableSAM2(finetuned_model_name = config.sam2.model_name, cfg = config.sam2.model_type, checkpoint = checkpoint_path, mode = "eval",
                              do_train_prompt_encoder = config.sam2.train_prompt_encoder, do_train_mask_decoder = config.sam2.train_mask_decoder,
                              img_embeddings_as_input = False, device = config.misc.device, weight_path = last_model_path)

        print('# Starting Without Post-Processing... #')
        scores_no_postpro = model.test_loop(dataloader, input_mask_eval = config.sam2.input_mask_eval, is_eval_post_processing = True, do_post_process = False)

        print('# Starting With Post-Processing... #')        
        scores_postpro = model.test_loop(dataloader, input_mask_eval = config.sam2.input_mask_eval, is_eval_post_processing = True, do_post_process = True, post_process_type = post_process_type)

    del model
    del dataloader
    torch.cuda.empty_cache()
    gc.collect()

    save_scores(scores_postpro, os.path.join(output_dir_path, f"scores_{testing_name}_postpro.json"), os.path.join(output_dir_path, f"avg_{testing_name}_postpro.json"))
    save_scores(scores_no_postpro, os.path.join(output_dir_path, f"scores_{testing_name}.json"), os.path.join(output_dir_path, f"avg_{testing_name}.json"))
