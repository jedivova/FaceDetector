import torch
import numpy as np
import torchgeometry as tgm
from mtcnn.utils.align_trans import get_reference_facial_points, get_transform_matrix, warp_and_crop_face

refrence = get_reference_facial_points(default_square= True)

def align_multi(img_tensor, landmarks, crop_size=(112, 112)):
    """Align muti-faces in a image
    
    Args:
        img_tensor (torch.Tensor)

        landmarks (np.ndarray or torch.IntTensor): Facial landmarks points with shape [n, 5, 2] 
    """

    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
        
    tr_matrixes = []
    for points in landmarks:
        matrix = get_transform_matrix(points, crop_size=crop_size, align_type='similarity')
        M = torch.from_numpy(matrix).unsqueeze(0).type(torch.float32)
        tr_matrixes.append(M)

    images = torch.cat([img_tensor]*len(tr_matrixes)).type(torch.float32)

    faces = tgm.warp_affine(images, torch.cat(tr_matrixes), dsize=(112, 112))

    return faces


def filter_side_face(boxes, landmarks):
    """Mask all side face judged through facial landmark points.
    
    Args:
        boxes (torch.IntTensor): Bounding boxes with shape [n, 4]
        landmarks (or torch.IntTensor): Facial landmarks points with shape [n, 5, 2]
    
    Returns:
        torch.Tensor: Tensor mask.
    """
    mid = (boxes[:, 2] + boxes[:, 0]).float() / 2
    mask =  (landmarks[:, 0, 0].float() - mid) * (landmarks[:, 1, 0].float() - mid) <= 0 

    return mask

    


