import numpy as np
import torch

from typing import Union

def collate_fn(batch : list[tuple[dict, np.ndarray]]):
    """
    Collate function for the torch DataLoader.

    Args:
        batch (list[tuple[dict, np.ndarray]]): list of data to batch

    Returns:
        (list, Tensor): batched data, in the form (data, mask)
    """
    batched_data = []
    batched_mask = []

    for data, mask in batch:
        batched_data.append(data)
        batched_mask.append(mask)

    return batched_data, torch.from_numpy(np.array(batched_mask))

def to_dict(img : Union[np.ndarray, dict], prompts : dict[str, np.ndarray], use_img_embeddings : bool = False, device : str = 'cuda', is_sam2_prompt : bool = False) -> dict:
    """
    Convert an element from an AbstractSAMDataset to a valid dictionnary regarding 
    Sam class or Sam2 class forward method specification.

    Args:
        img (Union[np.ndarray, dict]): the image (or embedding of that image)
        prompts (dict[str, np.ndarray]): the prompts for that image
        use_img_embeddings (bool, optional): whether the image is an embedding or not. Defaults to False.
        device (str, optional): the device to which the data must be transfered. Defaults to 'cuda'.
        is_sam2_prompt (bool, optional): boolean to tell if the formatting should be for sam or sam2 regarding the prompt. Defaults to False.

    Returns:
        dict: the formatted input to give to the model.
    """
    if use_img_embeddings:
        if isinstance(img, dict):
            processed_img = {k: v.to(device) if hasattr(v, "to") else v for k, v in img.items()}
        else:
            processed_img = img.to(device)

        original_size = (1024, 1024)
    else:
        processed_img = torch.from_numpy(img.copy()).permute(2, 0, 1).float().to(device)
        original_size = img.shape[:2]

    output = {'image': processed_img, 'original_size': original_size}
    # points
    if prompts['points'] is not None and prompts['neg_points'] is not None:
        if is_sam2_prompt is False:
            point_coords = torch.tensor(prompts['points']).float().to(device)
            point_labels = torch.ones(len(prompts['points'])).float().to(device)

            neg_point_coords = torch.tensor(prompts['neg_points']).float().to(device)
            neg_point_labels = torch.zeros(len(prompts['neg_points'])).float().to(device)

            output['point_coords'] = torch.cat([point_coords, neg_point_coords], dim=0).unsqueeze(0).to(device)
            output['point_labels'] = torch.cat([point_labels, neg_point_labels], dim=0).unsqueeze(0).to(device)

        else:
            point_labels = np.ones(len(prompts['points']), dtype = np.float32)
            neg_point_labels = np.zeros(len(prompts['neg_points']), dtype = np.float32)

            point_coords = prompts['points']
            neg_point_coords = prompts['neg_points']

            output['point_coords'] = np.concatenate([point_coords, neg_point_coords], axis = 0)
            output['point_labels'] = np.concatenate([point_labels, neg_point_labels], axis = 0)

    elif prompts['points'] is not None:
        if is_sam2_prompt is False:
            output['point_coords'] = torch.tensor(prompts['points']).float().unsqueeze(0).to(device)
            output['point_labels'] = torch.ones(len(prompts['points'])).float().unsqueeze(0).to(device)

        else:
            output['point_coords'] = prompts['points']
            output['point_labels'] = np.ones(len(prompts['points']), dtype = np.float32)

    elif prompts['neg_points'] is not None:
        if is_sam2_prompt is False:
            output['point_coords'] = torch.tensor(prompts['neg_points']).float().unsqueeze(0).to(device)
            output['point_labels'] = torch.zeros(len(prompts['neg_points'])).float().unsqueeze(0).to(device)

        else:
            output['point_coords'] = prompts['neg_points']
            output['point_labels'] = np.zeros(len(prompts['neg_points']), dtype = np.float32)
    # box
    if prompts['box'] is not None:
        if is_sam2_prompt is False:
            output['boxes'] = torch.tensor(prompts['box']).float().unsqueeze(0).to(device)

        else:
            output['boxes'] = prompts['box']

    if prompts['mask'] is not None:
        if is_sam2_prompt is False:
            output['mask_inputs'] = torch.tensor(prompts['mask']).float().unsqueeze(0).unsqueeze(0).to(device)

        else:
            output['mask_inputs'] = np.expand_dims(prompts['mask'], axis = 0).astype(np.float32)

    return output