from skimage.measure import *
from skimage import data
from scipy.stats import kurtosis, skew
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from skimage.io import imread
from skimage import io
import skimage
from skimage import measure
import pandas as pd

def extract_features(img_path):
    ds = []
    crr = []
    cn = []
    am = []
    en = []
    ho = []

    Class = []
    av_kur1 = []
    av_sk1 = []
    std = []
    Mean = []


    centroid = []
    area = []
    extent = []
    orientation = []
    convex_area = []
    solidity = []
    eccentricity = []
    euler_number = []
    im = io.imread(img_path)
    m = skimage.measure.moments(im)
    cr = m[0,1] / m[0,0]
    cc = m[1,0] / m[0,0]

    #measure.moments_central(im, cr, cc)

    label_img = label(im, connectivity=im.ndim)
    props = regionprops(label_img)
    area.append(props[0].area)
    extent.append(props[0].extent)
    orientation.append(props[0].orientation)
    convex_area.append(props[0].convex_area)
    solidity.append(props[0].solidity)
    eccentricity.append(props[0].eccentricity)
    euler_number.append(props[0].euler_number)
    glcm = greycomatrix(im, [5], [0], symmetric=True, normed=True)
    ds.append(greycoprops(glcm, 'dissimilarity')[0,0])
    crr.append(greycoprops(glcm, 'correlation')[0,0])
    cn.append(greycoprops(glcm, 'contrast')[0,0])
    am.append(greycoprops(glcm, 'ASM')[0,0])
    en.append(greycoprops(glcm, 'energy')[0,0])
    ho.append(greycoprops(glcm, 'homogeneity')[0,0])
    av_kur1.append(kurtosis(im.reshape(-1)))
    av_sk1.append(skew(im.reshape(-1)))
    Mean.append(im.mean())
    std.append(im.std())

    return pd.DataFrame({'area' : area,
                  'orientation' : orientation,
                  'convex_area' : convex_area,
                  'eccentricity' : eccentricity,
                  'dissimilarity' : ds,
                  'correlation' : crr,
                  'contrast' : cn,
                  'ASM' : am,
                  'energy' : en,
                  'homogeneity' : ho,
                  'kurtosis' : av_kur1,
                  'skew' : av_sk1,
                  'Standard Deviation' : std,
                  'Mean':Mean,
                  })

    # ip=list()
    # ip.append(float(area[0]))
    # ip.append(float(orientation[0]))
    # ip.append(float(convex_area[0]))
    # ip.append(float(eccentricity[0]))
    # ip.append(float(ds[0]))
    # ip.append(float(crr[0]))
    # ip.append(float(cn[0]))
    # ip.append(float(am[0]))
    # ip.append(float(en[0]))
    # ip.append(float(ho[0]))
    # ip.append(float(av_kur1[0]))
    # ip.append(float(av_sk1[0]))
    # ip.append(float(std[0]))
    # ip.append(float(Mean[0]))
    # return ip

