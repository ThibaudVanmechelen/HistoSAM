import numpy as np
import torch


def collate_fn(batch:list[tuple[dict, np.ndarray]]):
    '''Collate function for torch DataLoader. To use when creating a DataLoader'''
    batched_data = []
    batched_mask = []
    for data, mask in batch:
        batched_data.append(data)
        batched_mask.append(mask)
    return batched_data, torch.from_numpy(np.array(batched_mask))

def to_dict(img:np.ndarray, prompts:dict[str, np.ndarray], use_img_embeddings:bool=False, device:str='cuda') -> dict:
    '''Convert an element from an AbstractSAMDataset to a valid dictionnary regarding 
    Sam class forward method specification.'''
    if use_img_embeddings:
        processed_img = img.to(device)
        original_size = (1024, 1024)
    else:
        processed_img = torch.from_numpy(img.copy()).permute(2, 0, 1).float().to(device)
        original_size = img.shape[:2]
    output = {'image': processed_img, 'original_size': original_size}
    # points
    if prompts['points'] is not None and prompts['neg_points'] is not None:
        point_coords = torch.tensor(prompts['points']).float().to(device)
        point_labels = torch.ones(len(prompts['points'])).float().to(device)
        neg_point_coords = torch.tensor(prompts['neg_points']).float().to(device)
        neg_point_labels = torch.zeros(len(prompts['neg_points'])).float().to(device)
        output['point_coords'] = torch.cat([point_coords, neg_point_coords], dim=0).unsqueeze(0).to(device)
        output['point_labels'] = torch.cat([point_labels, neg_point_labels], dim=0).unsqueeze(0).to(device)
    elif prompts['points'] is not None:
        output['point_coords'] = torch.tensor(prompts['points']).float().unsqueeze(0).to(device)
        output['point_labels'] = torch.ones(len(prompts['points'])).float().unsqueeze(0).to(device)
    elif prompts['neg_points'] is not None:
        output['point_coords'] = torch.tensor(prompts['neg_points']).float().unsqueeze(0).to(device)
        output['point_labels'] = torch.zeros(len(prompts['neg_points'])).float().unsqueeze(0).to(device)
    # box
    if prompts['box'] is not None:
        output['boxes'] = torch.tensor(prompts['box']).float().unsqueeze(0).to(device)
    if prompts['mask'] is not None:
        output['mask_inputs'] = torch.tensor(prompts['mask']).float().unsqueeze(0).unsqueeze(0).to(device)
    return output