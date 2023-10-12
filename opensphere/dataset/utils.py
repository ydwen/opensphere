import random
import numpy as np
import cv2
import time

from PIL import Image
from skimage import transform
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def image_pipeline(info, test_mode):
    path = info['path']
    #image = Image.open(path).convert('RGB')
    image = cv2.imread(path)
    while image is None:
        time.sleep(10)
        image = cv2.imread(path)
    if image is None:
        raise OSError('{} is not found'.format(path))

    image = np.array(image)
    image = image[:, :, ::-1]

    # align the face image if the source and target landmarks are given
    src_landmark = info.get('src_landmark')
    tgz_landmark = info.get('tgz_landmark')
    crop_size = info.get('crop_size')
    if not (src_landmark is None or tgz_landmark is None or crop_size is None):
        tform = transform.SimilarityTransform()
        tform.estimate(tgz_landmark, src_landmark)
        M = tform.params[0:2, :]
        image = cv2.warpAffine(image, M, crop_size, borderValue=0.0)

    augmenter = info.get('augmenter')
    if not test_mode and augmenter is not None:
        image = augmenter.augment(image)
    if not test_mode and random.random() > 0.5:
        image = np.flip(image, axis=1).copy()

    # normalize to [-1, 1]
    image = ((image - 127.5) / 127.5)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)

    return image

def get_results(scores, labels, FPRs=[]):
    # roc curve
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    roc_curve = interp1d(fpr, tpr)
    # TPR @ FPR
    results = {f'TPR@FPR={FPR}': 100.*roc_curve(float(FPR)) for FPR in FPRs}
    # eer and auc
    results['EER'] = 100. * brentq(lambda x : 1. - x - roc_curve(x), 0., 1.)
    results['AUC'] = 100. * metrics.auc(fpr, tpr)
    # get acc
    tnr = 1. - fpr
    pos_num = labels.count(1)
    neg_num = labels.count(0)
    results['ACC'] = 100. * max(tpr * pos_num + tnr * neg_num) / len(labels)

    return results
