import os
from utils.config import load_config
from utils.save_scores import save_scores
from .evaluate import evaluate_standard_SAM_with_config, evaluate_without_prompts

def run_experiment_points(dataset_path : str, config_dir_path : str, checkpoint_path : str, output_dir_path : str, is_sam2 : bool):
    print("Loading the configs")
    config_1_point = load_config(os.path.join(config_dir_path, "prompting_1_point.toml"))
    config_5_point = load_config(os.path.join(config_dir_path, "prompting_5_point.toml"))

    print("Starting 1 point prompting")
    scores_1_point = evaluate_standard_SAM_with_config(config_1_point, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_1_point, os.path.join(output_dir_path, "scores_1_point.json"), os.path.join(output_dir_path, "avg_1_point.json"))

    print("Starting 5 point prompting")
    scores_5_point = evaluate_standard_SAM_with_config(config_5_point, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_5_point, os.path.join(output_dir_path, "scores_5_point.json"), os.path.join(output_dir_path, "avg_5_point.json"))


def run_experiment_box_with_points(dataset_path : str, config_dir_path : str, checkpoint_path : str, output_dir_path : str, is_sam2 : bool):
    print("Loading the configs")
    config_box = load_config(os.path.join(config_dir_path, "prompting_box.toml"))
    config_box_1_point = load_config(os.path.join(config_dir_path, "prompting_box_1_point_pos.toml"))
    config_box_5_point = load_config(os.path.join(config_dir_path, "prompting_box_5_point_pos.toml"))

    print("Starting box prompting")
    scores_box = evaluate_standard_SAM_with_config(config_box, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_box, os.path.join(output_dir_path, "scores_box.json"), os.path.join(output_dir_path, "avg_box.json"))

    print("Starting box with 1 point prompting")
    scores_box_1_point = evaluate_standard_SAM_with_config(config_box_1_point, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_box_1_point, os.path.join(output_dir_path, "scores_box_1_point.json"), os.path.join(output_dir_path, "avg_box_1_point.json"))

    print("Starting box with 5 point prompting")
    scores_box_5_point = evaluate_standard_SAM_with_config(config_box_5_point, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_box_5_point, os.path.join(output_dir_path, "scores_box_5_point.json"), os.path.join(output_dir_path, "avg_box_5_point.json"))


def run_experiment_box_with_neg_points(dataset_path : str, config_dir_path : str, checkpoint_path : str, output_dir_path : str, is_sam2 : bool):
    print("Loading the configs")
    config_box_1_point_in = load_config(os.path.join(config_dir_path, "prompting_box_1_point_neg_in.toml"))
    config_box_5_point_in = load_config(os.path.join(config_dir_path, "prompting_box_5_point_neg_in.toml"))
    config_box_1_point = load_config(os.path.join(config_dir_path, "prompting_box_1_point_neg_any.toml"))
    config_box_5_point = load_config(os.path.join(config_dir_path, "prompting_box_5_point_neg_any.toml"))

    print("Starting box with 1 negative point inside prompting")
    scores_1_point_in = evaluate_standard_SAM_with_config(config_box_1_point_in, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_1_point_in, os.path.join(output_dir_path, "scores_1_point_in.json"), os.path.join(output_dir_path, "avg_1_point_in.json"))

    print("Starting box with 5 negative points inside prompting")
    scores_5_point_in = evaluate_standard_SAM_with_config(config_box_5_point_in, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_5_point_in, os.path.join(output_dir_path, "scores_5_point_in.json"), os.path.join(output_dir_path, "avg_5_point_in.json"))

    print("Starting box with 1 negative point anywhere prompting")
    scores_1_point = evaluate_standard_SAM_with_config(config_box_1_point, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_1_point, os.path.join(output_dir_path, "scores_1_point.json"), os.path.join(output_dir_path, "avg_1_point.json"))

    print("Starting box with 5 negative point anywhere prompting")
    scores_5_point = evaluate_standard_SAM_with_config(config_box_5_point, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_5_point, os.path.join(output_dir_path, "scores_5_point.json"), os.path.join(output_dir_path, "avg_5_point.json"))


def run_experiment_mask(dataset_path : str, config_dir_path : str, checkpoint_path : str, output_dir_path : str, is_sam2 : bool):
    print("Loading the configs")
    config_gt = load_config(os.path.join(config_dir_path, "prompting_mask_ground_truth.toml"))
    config_scribble = load_config(os.path.join(config_dir_path, "prompting_mask_scribble.toml"))
    config_morphology = load_config(os.path.join(config_dir_path, "prompting_mask_morphology.toml"))
    config_loose_dilation = load_config(os.path.join(config_dir_path, "prompting_mask_loose_dilation.toml"))

    print("Starting ground truth mask prompting")
    scores_gt = evaluate_standard_SAM_with_config(config_gt, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_gt, os.path.join(output_dir_path, "scores_gt.json"), os.path.join(output_dir_path, "avg_gt.json"))

    print("Starting scribble mask prompting")
    scores_scribble = evaluate_standard_SAM_with_config(config_scribble, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_scribble, os.path.join(output_dir_path, "scores_scribble.json"), os.path.join(output_dir_path, "avg_scribble.json"))

    print("Starting morphology mask prompting")
    scores_morphology = evaluate_standard_SAM_with_config(config_morphology, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_morphology, os.path.join(output_dir_path, "scores_morphology.json"), os.path.join(output_dir_path, "avg_morphology.json"))

    print("Starting loose dilation mask prompting")
    scores_loose_dilation = evaluate_standard_SAM_with_config(config_loose_dilation, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_loose_dilation, os.path.join(output_dir_path, "scores_loose_dilation.json"), os.path.join(output_dir_path, "avg_loose_dilation.json"))


def run_experiment_mask_with_basic_prompts(dataset_path : str, config_dir_path : str, checkpoint_path : str, output_dir_path : str, is_sam2 : bool):
    print("Loading the configs")
    config_1_point_pos = load_config(os.path.join(config_dir_path, "prompting_mask_1_point_pos.toml"))
    config_5_point_pos = load_config(os.path.join(config_dir_path, "prompting_mask_5_point_pos.toml"))
    config_1_point_neg = load_config(os.path.join(config_dir_path, "prompting_mask_1_point_neg.toml"))
    config_5_point_neg = load_config(os.path.join(config_dir_path, "prompting_mask_5_point_neg.toml"))
    config_box = load_config(os.path.join(config_dir_path, "prompting_mask_box.toml"))

    print("Starting mask with 1 negative point prompting")
    scores_1_point_neg = evaluate_standard_SAM_with_config(config_1_point_neg, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_1_point_neg, os.path.join(output_dir_path, "scores_1_point_neg.json"), os.path.join(output_dir_path, "avg_1_point_neg.json"))

    print("Starting mask with 5 negative points prompting")
    scores_5_point_neg = evaluate_standard_SAM_with_config(config_5_point_neg, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_5_point_neg, os.path.join(output_dir_path, "scores_5_point_neg.json"), os.path.join(output_dir_path, "avg_5_point_neg.json"))

    print("Starting mask with 1 positive point prompting")
    scores_1_point_pos = evaluate_standard_SAM_with_config(config_1_point_pos, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_1_point_pos, os.path.join(output_dir_path, "scores_1_point_pos.json"), os.path.join(output_dir_path, "avg_1_point_pos.json"))

    print("Starting mask with 5 positive points prompting")
    scores_5_point_pos = evaluate_standard_SAM_with_config(config_5_point_pos, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_5_point_pos, os.path.join(output_dir_path, "scores_5_point_pos.json"), os.path.join(output_dir_path, "avg_5_point_pos.json"))

    print("Starting mask with box prompting")
    scores_box = evaluate_standard_SAM_with_config(config_box, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_box, os.path.join(output_dir_path, "scores_box.json"), os.path.join(output_dir_path, "avg_box.json"))


def run_experiment_all_prompts(dataset_path : str, config_dir_path : str, checkpoint_path : str, output_dir_path : str, is_sam2 : bool):
    print("Loading the configs")
    config_point_pos = load_config(os.path.join(config_dir_path, "prompting_mask_box_point_pos.toml"))
    config_point_neg = load_config(os.path.join(config_dir_path, "prompting_mask_box_point_neg.toml"))
    config_all = load_config(os.path.join(config_dir_path, "prompting_mask_all.toml"))

    print("Starting mask with box and negative point prompting")
    scores_point_neg = evaluate_standard_SAM_with_config(config_point_neg, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_point_neg, os.path.join(output_dir_path, "scores_point_neg.json"), os.path.join(output_dir_path, "avg_point_neg.json"))

    print("Starting mask with box and positive point prompting")
    scores_point_pos = evaluate_standard_SAM_with_config(config_point_pos, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_point_pos, os.path.join(output_dir_path, "scores_point_pos.json"), os.path.join(output_dir_path, "avg_point_pos.json"))

    print("Starting all prompting")
    scores_all = evaluate_standard_SAM_with_config(config_all, dataset_path, checkpoint_path, is_sam2)
    save_scores(scores_all, os.path.join(output_dir_path, "scores_all.json"), os.path.join(output_dir_path, "avg_all.json"))


def run_experiment_encoders_sam(dataset_path : str, config_dir_path : str, checkpoints : list[str], output_dir_path : str):
    print("Loading the configs")
    config_vit_h = load_config(os.path.join(config_dir_path, "prompting_vit_h.toml"))
    config_vit_l = load_config(os.path.join(config_dir_path, "prompting_vit_l.toml"))

    print("Starting prompting with ViT-L encoder")
    scores_vit_l = evaluate_standard_SAM_with_config(config_vit_l, dataset_path, checkpoints[0], False)
    save_scores(scores_vit_l, os.path.join(output_dir_path, "scores_vit_l.json"), os.path.join(output_dir_path, "avg_vit_l.json"))

    print("Starting prompting with ViT-H encoder")
    scores_vit_h = evaluate_standard_SAM_with_config(config_vit_h, dataset_path, checkpoints[1], False)
    save_scores(scores_vit_h, os.path.join(output_dir_path, "scores_vit_h.json"), os.path.join(output_dir_path, "avg_vit_h.json"))


def run_experiment_encoders_sam2(dataset_path : str, config_dir_path : str, checkpoints : list[str], output_dir_path : str):
    print("Loading the configs")
    config_hiera_t = load_config(os.path.join(config_dir_path, "prompting_hiera_t.toml"))
    config_hiera_s = load_config(os.path.join(config_dir_path, "prompting_hiera_s.toml"))
    config_hiera_l = load_config(os.path.join(config_dir_path, "prompting_hiera_l.toml"))

    print("Starting prompting with Hiera-Tiny encoder")
    scores_hiera_t = evaluate_standard_SAM_with_config(config_hiera_t, dataset_path, checkpoints[0], True)
    save_scores(scores_hiera_t, os.path.join(output_dir_path, "scores_hiera_t.json"), os.path.join(output_dir_path, "avg_hiera_t.json"))

    print("Starting prompting with Hiera-Small encoder")
    scores_hiera_s = evaluate_standard_SAM_with_config(config_hiera_s, dataset_path, checkpoints[1], True)
    save_scores(scores_hiera_s, os.path.join(output_dir_path, "scores_hiera_s.json"), os.path.join(output_dir_path, "avg_hiera_s.json"))

    print("Starting prompting with Hiera-Large encoder")
    scores_hiera_l = evaluate_standard_SAM_with_config(config_hiera_l, dataset_path, checkpoints[2], True)
    save_scores(scores_hiera_l, os.path.join(output_dir_path, "scores_hiera_l.json"), os.path.join(output_dir_path, "avg_hiera_l.json"))


def run_experiment_datasets(config_path : str, dataset_paths : list[str], checkpoint_path : str, output_dir_path : str, is_sam2 : bool):
    print("Loading the configs")
    config = load_config(config_path)

    for dataset in dataset_paths:
        dataset_name = os.path.basename(os.path.normpath(dataset))
        print(f"Starting prompting for dataset: {dataset_name}")

        scores = evaluate_standard_SAM_with_config(config, dataset, checkpoint_path, is_sam2)
        save_scores(scores, os.path.join(output_dir_path, f"{dataset_name}.json"), os.path.join(output_dir_path, f"avg_{dataset_name}.json"))


def run_experiment_no_prompts(dataset_path : str, model_config : list[(str, str)], output_dir_path : str, is_sam2 : bool, device : str):
    for cfg in model_config:
        checkpoint_path = cfg[0]
        model_type = cfg[1]

        print(f"Starting no prompts with {model_type}")
        scores = evaluate_without_prompts(dataset_path, checkpoint_path, is_sam2, model_type, device)

        if is_sam2 is False:
            model_name = model_type

        else:
            file_name = os.path.basename(model_type)
            model_name = file_name.split('.yaml')[0]

        save_scores(scores, os.path.join(output_dir_path, f"{model_name}.json"), os.path.join(output_dir_path, f"avg_{model_name}.json"))