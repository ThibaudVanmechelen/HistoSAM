import os
import gc
import torch

from torch.utils.data import DataLoader
from train import train_with_config, train_histo_sam_with_config
from model.sam2_model import TrainableSAM2
from model.model import load_model
from dataset_processing.dataset import SamDatasetFromFiles, filter_dataset, SAMDataset
from dataset_processing.preprocess import collate_fn
from evaluate import test_loop, evaluate_histo_SAM_with_config
from utils.config import load_config
from utils.save_scores import save_scores
from save_img_embeddings import save_embeddings
from save_img_embeddings import compute_embeddings_histo_sam

def file_verification(dataset_path, is_post_processing):
    """
    Function to verify if the dataset is 'in order' meaning that
    it does not already contain embeddings etc...

    Args:
        dataset_path (str): path to the dataset.
        is_post_processing (bool): whether the dataset should contain embeddings.
    """
    files_pre = {"mask.jpg", "img.jpg"}
    no_files_pre = {"prompt.pt", "img_embedding.pt", "sam2_img_embedding.pt"}
    files_post = {"mask.jpg", "img.jpg", "prompt.pt", "img_embedding.pt", "sam2_img_embedding.pt"}

    required_files = files_post if is_post_processing else files_pre
    listdir_ = os.listdir(dataset_path)
    print(f"Number of files: {len(listdir_)}")

    all_ok = True
    errors = []

    for subdir in listdir_:
        if os.path.isdir(os.path.join(dataset_path, subdir)):
            path = os.path.join(dataset_path, subdir)
            files_in_subdir = set(os.listdir(path))

            missing_files = required_files - files_in_subdir

            if is_post_processing:
                if missing_files:
                    all_ok = False
                    errors.append(f"Issue at path: {path}, Missing: {', '.join(missing_files)}")

            else:
                bad_files_present = files_in_subdir & no_files_pre
                if missing_files or bad_files_present:
                    all_ok = False

                    if missing_files:
                        errors.append(f"Issue at path: {path}, Missing: {', '.join(missing_files)}")

                    if bad_files_present:
                        errors.append(f"Issue at path: {path}, Forbidden files present: {', '.join(bad_files_present)}")

    if all_ok:
        print(f"All directories verified with success. Required files are present.")
    else:
        print("Some directories failed verification:")
        for error in errors:
            print(error)

        print("Error: redo procedure")


def run_embeddings(dataset_path : str, config_path : str, checkpoint_path_sam : str, checkpoint_path_sam2 : str):
    """
    Function to generate all the embeddings for a dataset.

    Args:
        dataset_path (str): path to the dataset.
        config_path (str): path to the config.
        checkpoint_path_sam (str): path to the sam checkpoint.
        checkpoint_path_sam2 (str):path to the sam2 checkpoint.
    """
    print("Loading the configs")
    config = load_config(config_path)

    print("Pre-processing file verification...")
    file_verification(os.path.join(dataset_path, 'processed'), is_post_processing = False)

    print("Beginning with SAM...")
    save_embeddings(config, dataset_path, checkpoint_path_sam, is_sam2 = False, save_prompt = True)

    print("Beginning with SAM2...")
    save_embeddings(config, dataset_path, checkpoint_path_sam2, is_sam2 = True, save_prompt = False)

    print("Post-processing file verification...")
    file_verification(os.path.join(dataset_path, 'processed'), is_post_processing = True)

    print("Done with embeddings !")


def run_finetuning(training_dataset_path : str, validation_dataset_path : str, config_path : str, checkpoint_path : str, is_sam2 : bool,
                   is_original_sam_loss: bool, output_dir_path : str, finetuning_name : str, use_dataset: list[bool]):
    """
    Function to perform the finetuning of a model (Sam or Sam2).

    Args:
        training_dataset_path (str): path to the training dataset.
        validation_dataset_path (str): path to the validation dataset.
        config_path (str): path to the config.
        checkpoint_path (str): path to the checkpoint.
        is_sam2 (bool): whether we finetune sam or sam2.
        is_original_sam_loss (bool): whether we use the original loss.
        output_dir_path (str): path to the output directory.
        finetuning_name (str): name of the finetuned model.
        use_dataset (list[bool]): which datasets should be used.
    """
    print("Loading the configs")
    config = load_config(config_path)

    if not is_sam2:
        print("Starting Training for SAM...")
        scores = train_with_config(config, checkpoint_path, training_dataset_path, validation_dataset_path, use_original_sam_loss = is_original_sam_loss, use_dataset = use_dataset)

    else:
        print("Starting Training for SAM2...")
        model = TrainableSAM2(finetuned_model_name = config.sam2.model_name, cfg = config.sam2.model_type, checkpoint = checkpoint_path, mode = "train",
                              do_train_prompt_encoder = config.sam2.train_prompt_encoder, do_train_mask_decoder = config.sam2.train_mask_decoder,
                              img_embeddings_as_input = config.training.use_img_embeddings, device = config.misc.device, weight_path = config.sam2.weight_path)

        scores = model.train_model(config, training_dataset_path, validation_dataset_path, use_original_sam_loss = is_original_sam_loss, use_dataset = use_dataset)

    save_scores(scores, os.path.join(output_dir_path, f"scores_{finetuning_name}.json"), os.path.join(output_dir_path, f"avg_{finetuning_name}.json"))


def run_finetuning_testing(dataset_path : str, config_path : str, checkpoint_path : str, is_sam2 : bool, output_dir_path : str, testing_name : str, use_dataset : list[bool], last_model_path : str = None):
    """
    Function to test the finetuned model.

    Args:
        dataset_path (str): path to the dataset.
        config_path (str): path to the config.
        checkpoint_path (str): path to the checkpoint.
        is_sam2 (bool): whether we test sam or sam2.
        output_dir_path (str): path to the output directory.
        testing_name (str): name of the file generated.
        use_dataset (list[bool]): which datasets should be used.
        last_model_path (str, optional): path for the weights to load. Defaults to None.
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
        use_img_embeddings = config.testing.use_img_embeddings,
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
        model = load_model(checkpoint_path, config.sam.model_type, img_embeddings_as_input = config.testing.use_img_embeddings, return_iou = True).to(config.misc.device)

        if last_model_path:
            print("Loading the last model...")
            model.load_state_dict(torch.load(last_model_path))

        scores = test_loop(model, dataloader, config.misc.device, config.sam.input_mask_eval, return_mean = False)

    else:
        print("Starting Testing for SAM2...")
        model = TrainableSAM2(finetuned_model_name = config.sam2.model_name, cfg = config.sam2.model_type, checkpoint = checkpoint_path, mode = "eval",
                              do_train_prompt_encoder = config.sam2.train_prompt_encoder, do_train_mask_decoder = config.sam2.train_mask_decoder,
                              img_embeddings_as_input = config.testing.use_img_embeddings, device = config.misc.device, weight_path = last_model_path)

        scores = model.test_loop(dataloader, input_mask_eval = config.sam2.input_mask_eval)

    del model
    del dataloader
    torch.cuda.empty_cache()
    gc.collect()

    save_scores(scores, os.path.join(output_dir_path, f"scores_{testing_name}.json"), os.path.join(output_dir_path, f"avg_{testing_name}.json"))


def run_finetuning_testing_per_prompt(dataset_path : str, config_path : str, checkpoint_path : str, is_sam2 : bool, output_dir_path : str, testing_name : str, use_dataset : list[bool], last_model_path : str = None):
    """
    Function to test the finetuned model, but gives back the metrics separated by prompt type.

    Args:
        dataset_path (str): path to the dataset.
        config_path (str): path to the config.
        checkpoint_path (str): path to the checkpoint.
        is_sam2 (bool): whether we test sam or sam2.
        output_dir_path (str): path to the output directory.
        testing_name (str): name of the file generated.
        use_dataset (list[bool]): which datasets should be used.
        last_model_path (str, optional): path for the weights to load. Defaults to None.
    """
    print("Loading the configs")
    config = load_config(config_path)

    print("Loading the model...")
    if not is_sam2:
        model = load_model(checkpoint_path, config.sam.model_type, img_embeddings_as_input = config.testing.use_img_embeddings, return_iou = True).to(config.misc.device)

        if last_model_path:
            print("Loading the last SAM model...")
            model.load_state_dict(torch.load(last_model_path))

    else:
        model = TrainableSAM2(finetuned_model_name = config.sam2.model_name, cfg = config.sam2.model_type, checkpoint = checkpoint_path, mode = "eval",
                            do_train_prompt_encoder = config.sam2.train_prompt_encoder, do_train_mask_decoder = config.sam2.train_mask_decoder,
                            img_embeddings_as_input = config.testing.use_img_embeddings, device = config.misc.device, weight_path = last_model_path)
        
    print("Creating the datasets...")
    prompt_type_array = [
        { 'points' : False, 'box' : False, 'neg_points' : False, 'mask' : True },
        { 'points' : False, 'box' : False, 'neg_points' : True, 'mask' : False },
        { 'points' : False, 'box' : True, 'neg_points' : False, 'mask' : False },
        { 'points' : True, 'box' : False, 'neg_points' : False, 'mask' : False }
    ]

    for prompt_type in prompt_type_array:
        current_prompt = [key for key, value in prompt_type.items() if value][0]
        print(f"Current prompt: {current_prompt}")

        dataset = SamDatasetFromFiles(
            root = dataset_path,
            transform = None,
            use_img_embeddings = config.testing.use_img_embeddings,
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
            scores = test_loop(model, dataloader, config.misc.device, config.sam.input_mask_eval, return_mean = False)

        else:
            scores = model.test_loop(dataloader, input_mask_eval = config.sam2.input_mask_eval)

        del dataloader
        torch.cuda.empty_cache()
        gc.collect()

        save_scores(scores, os.path.join(output_dir_path, f"scores_{testing_name}_{current_prompt}.json"), os.path.join(output_dir_path, f"avg_{testing_name}_{current_prompt}.json"))

    del model
    torch.cuda.empty_cache()
    gc.collect()


def run_recirculation_testing(dataset_path : str, config_path : str, checkpoint_path : str, is_sam2 : bool, output_dir_path : str, testing_name : str, use_dataset : list[bool], last_model_path : str = None):
    """
    Function to test the finetuned model, but here we recirculate the masks.

    Args:
        dataset_path (str): path to the dataset.
        config_path (str): path to the config.
        checkpoint_path (str): path to the checkpoint.
        is_sam2 (bool): whether we test sam or sam2.
        output_dir_path (str): path to the output directory.
        testing_name (str): name of the file generated.
        use_dataset (list[bool]): which datasets should be used.
        last_model_path (str, optional): path for the weights to load. Defaults to None.
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
        use_img_embeddings = config.testing.use_img_embeddings,
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
        model = load_model(checkpoint_path, config.sam.model_type, img_embeddings_as_input = config.testing.use_img_embeddings, return_iou = True).to(config.misc.device)

        if last_model_path:
            print("Loading the last model...")
            model.load_state_dict(torch.load(last_model_path))

        scores = test_loop(model, dataloader, config.misc.device, config.sam.input_mask_eval, return_mean = False, do_recirculation = True)

    else:
        print("Starting Testing for SAM2...")
        model = TrainableSAM2(finetuned_model_name = config.sam2.model_name, cfg = config.sam2.model_type, checkpoint = checkpoint_path, mode = "eval",
                              do_train_prompt_encoder = config.sam2.train_prompt_encoder, do_train_mask_decoder = config.sam2.train_mask_decoder,
                              img_embeddings_as_input = config.testing.use_img_embeddings, device = config.misc.device, weight_path = last_model_path)

        scores = model.test_loop(dataloader, input_mask_eval = config.sam2.input_mask_eval, do_recirculation = True)

    del model
    del dataloader
    torch.cuda.empty_cache()
    gc.collect()

    save_scores(scores, os.path.join(output_dir_path, f"scores_{testing_name}_recirculation.json"), os.path.join(output_dir_path, f"avg_{testing_name}_recirculation.json"))


def run_mask_distribution_experiment(dataset_path : str, config_path : str, checkpoint_path : str, is_sam2 : bool, output_dir_path : str, testing_name : str, last_model_path : str = None):
    """
    Function to run experiment to see if the mask distribution influences the performances.

    Args:
        dataset_path (str): path to the dataset.
        config_path (str): path to the config.
        checkpoint_path (str): path to the checkpoint.
        is_sam2 (bool): whether we test sam or sam2.
        output_dir_path (str): path to the output directory.
        testing_name (str): name of the file generated.
        last_model_path (str, optional): path for the weights to load. Defaults to None.
    """      
    print("Loading the configs")
    config = load_config(config_path)

    print("Creating the dataset...")
    prompt_type = {
        'points' : False,
        'box' : False,
        'neg_points' : False,
        'mask' : True
    }

    dataset_loose_dilation = SAMDataset(
        root = dataset_path,
        prompt_type = prompt_type,
        n_points = 0,
        n_neg_points = 0,
        verbose = True,
        to_dict = True,
        use_img_embeddings = False,
        neg_points_inside_box = False,
        points_near_center = 4,
        random_box_shift = 20,
        mask_prompt_type = 'loose_dilation',
        box_around_mask = False,
        is_sam2_prompt = is_sam2
    )

    dataset_scribble = SAMDataset(
        root = dataset_path,
        prompt_type = prompt_type,
        n_points = 0,
        n_neg_points = 0,
        verbose = True,
        to_dict = True,
        use_img_embeddings = False,
        neg_points_inside_box = False,
        points_near_center = 4,
        random_box_shift = 20,
        mask_prompt_type = 'scribble',
        box_around_mask = False,
        is_sam2_prompt = is_sam2
    )

    dataloader_scribble = DataLoader(dataset_scribble, batch_size = config.testing.batch_size, shuffle = False, collate_fn = collate_fn)
    dataloader_loose_dilation = DataLoader(dataset_loose_dilation, batch_size = config.testing.batch_size, shuffle = False, collate_fn = collate_fn)

    if not is_sam2:
        print("Starting Testing for SAM...")
        model = load_model(checkpoint_path, config.sam.model_type, img_embeddings_as_input = False, return_iou = True).to(config.misc.device)

        if last_model_path:
            print("Loading the last model...")
            model.load_state_dict(torch.load(last_model_path))

        scores_scribble = test_loop(model, dataloader_scribble, config.misc.device, config.sam.input_mask_eval, return_mean = False)
        scores_loose_dilation = test_loop(model, dataloader_loose_dilation, config.misc.device, config.sam.input_mask_eval, return_mean = False)

    else:
        print("Starting Testing for SAM2...")
        model = TrainableSAM2(finetuned_model_name = config.sam2.model_name, cfg = config.sam2.model_type, checkpoint = checkpoint_path, mode = "eval",
                              do_train_prompt_encoder = config.sam2.train_prompt_encoder, do_train_mask_decoder = config.sam2.train_mask_decoder,
                              img_embeddings_as_input = False, device = config.misc.device, weight_path = last_model_path)

        scores_scribble = model.test_loop(dataloader_scribble, input_mask_eval = config.sam2.input_mask_eval)
        scores_loose_dilation = model.test_loop(dataloader_loose_dilation, input_mask_eval = config.sam2.input_mask_eval)

    del model
    del dataloader_scribble
    del dataloader_loose_dilation
    torch.cuda.empty_cache()
    gc.collect()

    save_scores(scores_scribble, os.path.join(output_dir_path, f"scores_{testing_name}_scribble.json"), os.path.join(output_dir_path, f"avg_{testing_name}_scribble.json"))
    save_scores(scores_loose_dilation, os.path.join(output_dir_path, f"scores_{testing_name}_ld.json"), os.path.join(output_dir_path, f"avg_{testing_name}_ld.json"))


def run_finetuning_histoSAM(training_dataset_path : str, validation_dataset_path : str, config_path : str, checkpoint_paths : list[str],
                   is_original_sam_loss: bool, output_dir_path : str, finetuning_name : str, use_dataset: list[bool]):
    """
    Function to finetune histoSAM model.

    Args:
        training_dataset_path (str): path to the training dataset.
        validation_dataset_path (str): path to the validation dataset.
        config_path (str): path to the config.
        checkpoint_paths (list[str]): list of checkpoints for the model. Must have size 2, 0 = SAM, 1 = Histo encoder
        is_original_sam_loss (bool): whether to use the original loss of sam.
        output_dir_path (str): path to the output directory.
        finetuning_name (str): name of the finetuned model.
        use_dataset (list[bool]): which datasets should be used.
    """
    print("Loading the config...")
    config = load_config(config_path)

    print("Computing the image embeddings for training set...")
    compute_embeddings_histo_sam(config, training_dataset_path, checkpoint_paths)

    print("Computing the image embeddings for validation set...")
    compute_embeddings_histo_sam(config, validation_dataset_path, checkpoint_paths)

    print("Starting Training for HistoSAM...")
    scores = train_histo_sam_with_config(config, checkpoint_paths, training_dataset_path, validation_dataset_path, is_original_sam_loss, use_dataset)

    save_scores(scores, os.path.join(output_dir_path, f"scores_{finetuning_name}.json"), os.path.join(output_dir_path, f"avg_{finetuning_name}.json"))


def run_finetuning_testing_histoSAM(dataset_path : str, config_path : str, checkpoint_paths : list[str], output_dir_path : str, 
                                    testing_name : str, use_dataset : list[bool], last_model_path : str, encoder_type : str, deconv : bool,
                                    fuse_with_attention : bool = False, refine_with_attention : bool = False, sam_weights_for_refinement : str = None):
    """
    Function to test the finetuned histoSAM model.

    Args:
        dataset_path (str): path to the dataset.
        config_path (str): path to the config.
        checkpoint_paths (list[str]): list of checkpoints for the model. Must have size 2, 0 = SAM, 1 = Histo encoder
        output_dir_path (str): path to the output directory.
        testing_name (str): name of the testing files.
        use_dataset (list[bool]): which datasets should be used.
        last_model_path (str): path to the trained weights.
        encoder_type (str): type of the encoder.
        deconv (bool): whether to use deconv or efficient upsampling.
        fuse_with_attention (bool, optional): whether to fuse with attention. Defaults to False.
        refine_with_attention (bool, optional): wehter to refine with attention. Defaults to False.
        sam_weights_for_refinement (str, optional): path to the weights for refinement if need to refine. Defaults to None.
    """
    print("Loading the config...")
    config = load_config(config_path)

    print("Computing the image embeddings for testing set...")
    compute_embeddings_histo_sam(config, dataset_path, checkpoint_paths, encoder_type, r_w_a = refine_with_attention, sam_weights_for_refinement = sam_weights_for_refinement)

    print("Evaluating HistoSAM...")
    scores = evaluate_histo_SAM_with_config(config, dataset_path, checkpoint_paths, use_dataset, 
                                            last_model_path, encoder_type, deconv, fuse_with_attention, refine_with_attention,
                                            sam_weights_for_refinement)

    save_scores(scores, os.path.join(output_dir_path, f"scores_{testing_name}.json"), os.path.join(output_dir_path, f"avg_{testing_name}.json"))