"""Save img embeddings in file for further use. Allows to train SAM model without touching its image encoder"""
import os

import gc
import torch
from dataset_processing.dataset import SAMDataset, SamDatasetFromFiles
from model.model import load_model
from model.sam2_model import TrainableSAM2
from model.histo_sam import HistoSAM
from tqdm import tqdm

def save_embeddings(config : dict, dataset_path : str, checkpoint_path : str, is_sam2 : bool, save_prompt : bool):
    """
    Function to save img embeddings in file for further use. Allows to train SAM model without touching its image encoder

    Args:
        config (dict): the config.
        dataset_path (str): path to the dataset.
        checkpoint_path (str): path to the checkpoint.
        is_sam2 (bool): whether we compute the embeddings for sam or sam2.
        save_prompt (bool): whether we also save the prompts or not.
    """
    if not is_sam2:
        model = load_model(checkpoint_path, config.sam.model_type).to(device = config.misc.device)
    else:
        model = TrainableSAM2(finetuned_model_name = "embedding_model", cfg = config.sam2.model_type, checkpoint = checkpoint_path, mode = "eval",
                              do_train_prompt_encoder = False, do_train_mask_decoder = False, img_embeddings_as_input = False, device = config.misc.device)

    dataset = SAMDataset(
        root = dataset_path,
        prompt_type = {'points' : True, 'box' : True, 'neg_points' : True, 'mask' : True},
        n_points = config.dataset.n_points,
        n_neg_points = config.dataset.n_neg_points,
        verbose = True,
        to_dict = True,
        use_img_embeddings = False,
        neg_points_inside_box = config.dataset.negative_points_inside_box,
        points_near_center = config.dataset.points_near_center,
        random_box_shift = config.dataset.random_box_shift,
        mask_prompt_type = config.dataset.mask_prompt_type,
        box_around_mask = config.dataset.box_around_prompt_mask,
        is_sam2_prompt = is_sam2,
        is_embedding_saving = True
    )
    
    if not is_sam2:
        model.eval()
    
    with torch.no_grad():
        for i, (data, _, prompt) in tqdm(enumerate(dataset), total = len(dataset), desc = 'Saving img embeddings'):
            if not is_sam2:
                img_embedding = model.get_image_embeddings([data])
                img_embedding_file_name = 'img_embedding.pt'
            else:
                sam2_image = model.convert_img_from_sam_to_sam2_format(data['image'])
                model.set_image(sam2_image)
                img_embedding = model._features
                img_embedding_file_name = 'sam2_img_embedding.pt'

            file_name = dataset.images[i]
            img_dir = os.path.dirname(file_name) # here save the embedding in the same dir as the img

            img_save_path = os.path.join(img_dir, img_embedding_file_name)
            torch.save(img_embedding, img_save_path)

            if save_prompt:
                prompt_save_path = os.path.join(img_dir, 'prompt.pt')
                torch.save(prompt, prompt_save_path)

    del model
    del dataset
    torch.cuda.empty_cache()
    gc.collect()


def save_prompts(config : dict, dataset_path : str, is_sam2 : bool):
    """
    Function to save only the prompts for training.

    Args:
        config (dict): the config.
        dataset_path (str): path to the dataset.
        is_sam2 (bool): whether we compute the embeddings for sam or sam2.
    """
    dataset = SamDatasetFromFiles(
        root = dataset_path,
        prompt_type = {'points' : True, 'box' : True, 'neg_points' : True, 'mask' : True},
        n_points = config.dataset.n_points,
        n_neg_points = config.dataset.n_neg_points,
        verbose = True,
        to_dict = True,
        use_img_embeddings = False,
        neg_points_inside_box = config.dataset.negative_points_inside_box,
        points_near_center = config.dataset.points_near_center,
        random_box_shift = config.dataset.random_box_shift,
        mask_prompt_type = config.dataset.mask_prompt_type,
        box_around_mask = config.dataset.box_around_prompt_mask,
        is_sam2_prompt = is_sam2,
        is_embedding_saving = True,
        load_on_cpu = False,
        generate_prompt_on_get = True
    )

    for i, (_, _, prompt) in tqdm(enumerate(dataset), total = len(dataset), desc = 'Saving prompt embeddings'):
        file_name = dataset.images[i]
        img_dir = os.path.dirname(file_name)

        prompt_save_path = os.path.join(img_dir, 'prompt.pt')
        torch.save(prompt, prompt_save_path)

    del dataset


def compute_embeddings_histo_sam(config : dict, dataset_path : str, checkpoint_paths : list, encoder_type : str = None, 
                                 r_w_a = None, sam_weights_for_refinement = None):
    """
    Function to compute the embeddings for histoSAM.

    Args:
        config (dict): the config.
        dataset_path (str): path to the dataset.
        checkpoint_paths (list[str]): list of checkpoints for the model. Must have size 2, 0 = SAM, 1 = Histo encoder
        encoder_type (str, optional): type of the encoder. Defaults to None.
        r_w_a (optional): whether to use refine with attention or not. Defaults to None.
        sam_weights_for_refinement (optional): path to the weights if we use refine with attention. Defaults to None.
    """
    r_w_a = config.encoder.get("refine_with_attention", False) if r_w_a is None else r_w_a
    sam_weights_for_refinement = config.encoder.get("sam_weights_for_refinement", None) if sam_weights_for_refinement is None else sam_weights_for_refinement

    model = HistoSAM(
        model_type = config.sam.model_type,
        checkpoint_path = checkpoint_paths[0],
        hist_encoder_type = config.encoder.type if encoder_type == None else encoder_type,
        hist_encoder_checkpoint_path = checkpoint_paths[1],
        not_use_sam_encoder = config.sam.not_use_sam_encoder,
        embedding_as_input = False,
        up_sample_with_deconvolution = False,
        freeze_sam_img_encoder = True,
        freeze_prompt_encoder = True,
        freeze_mask_decoder = True,
        return_iou = True,
        device = config.misc.device,                  
        refine_with_attention = r_w_a,
        sam_weights_for_refinement = sam_weights_for_refinement      
    )

    dataset = SAMDataset(
        root = dataset_path,
        prompt_type = {'points' : False, 'box' : False, 'neg_points' : False, 'mask' : False},
        n_points = 0,
        n_neg_points = 0,
        verbose = True,
        to_dict = True,
        use_img_embeddings = False,
        neg_points_inside_box = False,
        points_near_center = 0,
        random_box_shift = 0,
        mask_prompt_type = 'truth',
        box_around_mask = False,
        is_sam2_prompt = False,
        is_embedding_saving = True
    )

    model.compute_all_img_embeddings(dataset)

    del model
    del dataset
    torch.cuda.empty_cache()
    gc.collect()


def remove_pt_files(dataset_path):
    """
    Function to remove the .pt files that were saved in the dataset.

    Args:
        dataset_path (str): path to the dataset where we remove the files.
    """
    pt_files = {"prompt.pt", "img_embedding.pt", "sam2_img_embedding.pt"}

    listdir_ = os.listdir(dataset_path)
    print(f"Number of files: {len(listdir_)}")

    for subdir in listdir_:
        if os.path.isdir(os.path.join(dataset_path, subdir)):
            path = os.path.join(dataset_path, subdir)

            for file in os.listdir(path):
                if file in pt_files:
                    file_path = os.path.join(path, file)

                    try:
                        os.remove(file_path)

                    except Exception as e:
                        print(f"Error removing file {file_path}: {e}")

    print("Done with removal !")