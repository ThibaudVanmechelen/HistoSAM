import os

from train import train_with_config
from model.sam2_model import TrainableSAM2
from utils.config import load_config
from utils.save_scores import save_scores
from save_img_embeddings import save_embeddings

def file_verification(dataset_path, is_post_processing):
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

            if is_post_processing and missing_files:
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


def run_finetuning(training_dataset_path : str, validation_dataset_path : str, config_path : str, checkpoint_path : str, is_sam2 : str, 
                   is_original_sam_loss: bool, output_dir_path : str, finetuning_name : str):
    print("Loading the configs")
    config = load_config(config_path)

    print("Starting Training...")
    if not is_sam2:
        scores = train_with_config(config, checkpoint_path, training_dataset_path, validation_dataset_path, use_original_sam_loss = is_original_sam_loss)

    else:
        model = TrainableSAM2(finetuned_model_name = config.sam2.model_name, cfg = config.sam2.model_type, checkpoint = checkpoint_path, mode = "train",
                              do_train_prompt_encoder = config.sam2.train_prompt_encoder, do_train_mask_decoder = config.sam2.train_mask_decoder, 
                              img_embeddings_as_input = config.sam2.img_embeddings_as_input, device = config.misc.device, weight_path = config.sam2.weight_path)

        scores = model.train_model(config, training_dataset_path, validation_dataset_path, use_original_sam_loss = is_original_sam_loss)

    save_scores(scores, os.path.join(output_dir_path, f"scores_{finetuning_name}.json"), os.path.join(output_dir_path, f"avg_{finetuning_name}.json")) # TODO check if save_scores will work


def run_finetuning_testing(): # TODO
    return