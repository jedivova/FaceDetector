import torch
import mtcnn
from mtcnn.deploy.align import align_multi
import insightface



def get_embeddings(img_path, detector, embedder):
    """Align muti-faces in a image

    Args:
        @img_path: str, path to the image
        @detector: mtcnn faces detecor
        @embedder: faces embedder

    Returns:
    ----------
        @features: torch.Tensor [N,512], features of all detected faces
    """

    img_tensor = detector._preprocess(img_path)
    boxes, landmarks = detector.detect(img_tensor)
    outp = align_multi(img_tensor, landmarks, crop_size=(112, 112))

    with torch.no_grad():
        features = embedder(outp)
    return features


def get_models(mtcnn_w_path='mtcnn/weights'):
    # First we create pnet, rnet, onet, and load weights from caffe model.
    pnet, rnet, onet = mtcnn.get_net_caffe(mtcnn_w_path)

    # Then we create a detector
    detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cpu')

    embedder = insightface.iresnet100(pretrained=True)
    embedder.eval()

    return detector, embedder



if __name__=='__main__':
    detector, embedder = get_models()

    img_path = 'tests/asset/images/roate.jpg'
    features = get_embeddings(img_path, detector, embedder)
    print(features.shape)
    print(features[0,:5])
