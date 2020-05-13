# %% [code]
import itertools
from PIL import Image, ImageOps
import numpy as np
import os
import shutil
import gc
import tensorflow
from tensorflow.keras.applications import vgg16, inception_resnet_v2, nasnet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Lambda, Concatenate, Input
import random


DATA_PATH = "./dataset/"

#========================= UTILS =========================#

def setup(src, dst, supp_img):

    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    if os.path.exists(supp_img):
        shutil.rmtree(supp_img)
    os.mkdir(supp_img)


def load_thumb_pad(img_path, size):
    size_t = (size, size)
    des_size = size
    img = Image.open(img_path)

    # thumb
    img.thumbnail(size_t)
    # pad
    tw, th = img.size
    dw = des_size - tw
    dh = des_size - th
    pad = (dw//2, dh//2, dw-(dw//2), dh-(dh//2))
    new_img = ImageOps.expand(img, pad)
    return np.array(new_img)

'''This function generate the feature vectors for the dataset specified as first input,
the representative(s) for each class are randomly chosen accordingly to a seed'''

def gen_support_set(dataset_path, n_shot, seed_val, feat_extr_name, vector_file):

    random.seed(seed_val)
    model, preprocessor, img_size = load_feat_extr(feat_extr_name)

    image_names = os.listdir(dataset_path)
    num_vecs = 0

    image_groups = {}
    for image_name in image_names:
        name_split = image_name.split("_")
        label = name_split[0]  # group name

        if label in image_groups.keys():
            image_groups[label] += [image_name]
        else:
            image_groups[label] = [image_name, ]

    fvec = open(vector_file, "w")

    support_list = []
    for key in image_groups.keys():
        support_list += random.sample(set(image_groups[key]), n_shot)

    support_np = []
    for image_name in support_list:
        image = load_thumb_pad(DATA_PATH+image_name, img_size)
        os.rename(os.path.join(DATA_PATH+image_name), os.path.join("./support_set_images/", image_name))
        support_np.append(image)

    # print(len(batched_images))
    X = preprocessor(np.array(support_np, dtype="float32"))
    vectors = model.predict(X)

    for i in range(vectors.shape[0]):
        image_vector = ",".join(["{:.5e}".format(v) for v in vectors[i].tolist()])
        fvec.write("{:s}\t{:s}\n".format(support_list[i], image_vector))
        num_vecs += 1
    print("{:d} vectors generated".format(num_vecs))
    fvec.close()

    del model, preprocessor, img_size, X, vectors, support_np, support_list
    gc.collect()


def load_feat_extr(fname):
    model = None
    preprocessor = None
    img_size = -1

    if fname == "nasnet":
        feat_model = nasnet.NASNetLarge(weights = None, include_top=True)
        feat_model.load_weights("/kaggle/input/keras-pretrain-model-weights/NASNet-large.h5")
        preprocessor = nasnet.preprocess_input
        #feat_model.summary()
        model = Model(inputs=feat_model.input, outputs=feat_model.layers[-2].output)
        img_size = 331

    return model, preprocessor, img_size


def load_classifier(model_path):
    model = tensorflow.keras.models.load_model(model_path)
    #model.summary()

    return model

def classify_image(imgs_vec, vec_groups, model):

    pred = []

    for key in sorted(vec_groups.keys()):

        X1 = np.vstack([np.tile(img_vec, (len(vec_groups[key]), 1)) for img_vec in imgs_vec])
        X2 = np.vstack([vec_groups[key] for _ in range(len(imgs_vec))])

        Y_pred = model.predict([X1, X2])

        yes = Y_pred[:,1]
        yes = [np.mean(yes[index * len(vec_groups[key]): (index+1) * len(vec_groups[key])]) for index in range(len(imgs_vec))]

        yes = [np.round(y*100, 3) for y in yes]

        pred += [yes]


    pred = np.vstack(pred)
    return np.argmax(pred, axis = 0)



def eval(model, dataset, support_vecs, feat_extr_name, batch_size):

    feat_extr, preprocessor, img_size = load_feat_extr(feat_extr_name)

    vec_groups = {}

    fvec = open(support_vecs, "r")
    for line in fvec:
        image_name, image_vec = line.strip().split("\t")
        vec = np.array([float(v) for v in image_vec.split(",")])
        label = int(image_name.split("_")[0])

        if label in vec_groups.keys():
            vec_groups[label] += [vec]
        else:
            vec_groups[label] = [vec, ]

    fvec.close()

    all_images = os.listdir(dataset)

    right_pred = 0

    imgs_resized = []
    true_labels = []

    for index, image in enumerate(all_images):
        imgs_resized.append(load_thumb_pad(os.path.join(dataset, image), img_size))
        true_labels.append(int(image.split("_")[0]))

    for index in range((len(imgs_resized)//batch_size)+1):

        X = preprocessor(np.array(imgs_resized[index*batch_size : (index+1)*batch_size], dtype="float32"))
        vec_batch = feat_extr.predict(X)

        batch_labels = true_labels[index*batch_size : (index+1)*batch_size]

        pred_labels = classify_image(vec_batch, vec_groups, model)

        print(str((index+1)*batch_size) + " images processed out of " + str(len(all_images)))

        for index, _ in enumerate(batch_labels):
            if batch_labels[index] == pred_labels[index]:
                right_pred += 1

        del X, vec_batch
        gc.collect()

    print("Right predictions: " + str(right_pred) + " out of " + str(len(all_images)) + " Accuracy: " + str((right_pred/len(all_images))*100) )


model = load_classifier("/kaggle/input/weights-fcn/nesnet_flowers_9938.h5")
setup("/kaggle/input/oxford102/flowers/", DATA_PATH, "support_set_images")
gen_support_set("./dataset/", 1, 8, "nasnet", "nasnet.tsv")
eval(model, "./dataset", "nasnet.tsv", "nasnet", 64)
