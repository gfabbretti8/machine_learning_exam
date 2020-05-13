# %% [code]
import itertools
import shutil
from PIL import Image, ImageOps
import numpy as np
import os
from keras import backend as K
import tensorflow
from tensorflow.keras.applications import vgg16, inception_resnet_v2, nasnet, mobilenet_v2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Lambda, Concatenate, Input, GlobalMaxPooling2D
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import imgaug as ia
import imgaug.augmenters as iaa

# %% [markdown]
# # Utils

# %% [code]
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
    return np.asarray(new_img)


def image_batch_generator(image_names, batch_size):
    num_batches = len(image_names) // batch_size
    for i in range(num_batches):
        batch = image_names[i * batch_size: (i + 1) * batch_size]
        yield batch
        # attenzione!
    batch = image_names[(i + 1) * batch_size:]
    yield batch


def get_images_transformers():

    return []

def vectorize_images(image_dir, image_size, preprocessor,model, vector_file, sample_per_class, batch_size=32):

    image_names = os.listdir(image_dir)
    num_vecs = 0

    image_names = [list(v)[:sample_per_class] for k,v in itertools.groupby(sorted(image_names),key=lambda x: x.split("_")[0] )]

    image_names = list(itertools.chain(*image_names)) #to flatten the list

    transformations = get_images_transformers()

    fvec = open(vector_file, "w")
    for image_batch in image_batch_generator(image_names, batch_size):
        if len(image_batch) == 0: break  # orribile, ma serve per gestire l'ultima iterazione, altrimenti salta
        batched_images = []

        for image_name in image_batch:
            image = load_thumb_pad(image_dir+image_name, image_size)
            batched_images.append(image)

        image_batch_augmented = []
        batched_images_augmented = []

        for i, img in enumerate(batched_images):
            img_name = image_batch[i]
            image_batch_augmented.append(img_name) #adding the image without augmentation
            batched_images_augmented.append(img)

            for i, transform in enumerate(transformations):
                img_augmented = transform.augment_image(img)
                img_name_augmented = img_name.split(".")[0] + str(i) + ".jpg"
                image_batch_augmented.append(img_name_augmented) #adding the image without augmentation
                batched_images_augmented.append(img_augmented)


        # print(len(batched_images))
        X = preprocessor(np.array(batched_images_augmented, dtype="float32"))
        vectors = model.predict(X)
        # print(vectors.shape)
        for i in range(vectors.shape[0]):
            if num_vecs % 100 == 0:
                print("{:d} vectors generated".format(num_vecs))
            image_vector = ",".join(["{:.5e}".format(v) for v in vectors[i].tolist()])
            fvec.write("{:s}\t{:s}\n".format(image_batch_augmented[i], image_vector))
            num_vecs += 1
    print("{:d} vectors generated".format(num_vecs))
    fvec.close()

def get_triples(vectors_file):
    assert os.path.exists(vectors_file)

    image_groups = {}

    fvec = open(vectors_file, "r")
    for line in fvec:
        image_name, image_vec = line.strip().split("\t")
        name_split = image_name.split("_")
        class_name = name_split[0]  # group name
        # id_img = name_split[1]  # base name

        if class_name in image_groups.keys():
            image_groups[class_name] += [image_name]
        else:
            image_groups[class_name] = [image_name, ]
    fvec.close()

    num_sim = 0
    image_triples = []
    group_list = sorted(list(image_groups.keys()))

    for i, g in enumerate(group_list):
        if num_sim % 100 == 0:
            print("Generated {:d} pos + {:d} neg = {:d} total image triples"
                  .format(num_sim, num_sim, 2 * num_sim), end="\r")

        images_in_group = image_groups[g]
        # generate similar pairs
        sim_pairs_iter = itertools.combinations(images_in_group, 2)

        # for each similar pair, generate a different pair
        for ref_image, sim_image in sim_pairs_iter:
            image_triples.append((ref_image, sim_image, 1))
            num_sim += 1

            # sceglie classe random (esclude la classe g già in uso nel ciclo corrente)
            while True:
                j = np.random.randint(low=0, high=len(group_list), size=1)[0]
                if j != i: break

            dif_image_candidates = image_groups[group_list[j]]
            # sceglie l'immagine
            k = np.random.randint(low=0, high=len(dif_image_candidates), size=1)[0]
            dif_image = dif_image_candidates[k]
            image_triples.append((ref_image, dif_image, 0))

    print("Generated {:d} pos + {:d} neg = {:d} total image triples, COMPLETE"
          .format(num_sim, num_sim, 2 * num_sim))
    return image_triples


def load_vectors(vector_file):
    vec_dict = {}
    fvec = open(vector_file, "r")
    for line in fvec:
        image_name, image_vec = line.strip().split("\t")
        vec = np.array([float(v) for v in image_vec.split(",")])
        vec_dict[image_name] = vec
    fvec.close()
    return vec_dict


def train_test_split(triples, splits):
    assert sum(splits) == 1.0
    split_pts = np.cumsum(np.array([0.] + splits))
    indices = np.random.permutation(np.arange(len(triples)))
    shuffled_triples = [triples[i] for i in indices]
    data_splits = []
    for sid in range(len(splits)):
        start = int(split_pts[sid] * len(triples))
        end = int(split_pts[sid + 1] * len(triples))
        data_splits.append(shuffled_triples[start:end])
    return data_splits


def batch_to_vectors(batch, vec_size, vec_dict):
    X1 = np.zeros((len(batch), vec_size))
    X2 = np.zeros((len(batch), vec_size))
    Y = np.zeros((len(batch), 2))
    for tid in range(len(batch)):
        X1[tid] = vec_dict[batch[tid][0]]
        X2[tid] = vec_dict[batch[tid][1]]
        #adapted to sigmoid (maybe)
        Y[tid] = 0 if batch[tid][2] == 0 else 1
    return [X1, X2], Y


def data_generator(triples, vec_size, vec_dict, batch_size=32):
    while True:
        # shuffle once per batch
        indices = np.random.permutation(np.arange(len(triples)))
        num_batches = len(triples) // batch_size
        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
            batch = [triples[i] for i in batch_indices]
            yield batch_to_vectors(batch, vec_size, vec_dict)


def get_model(VECTOR_SIZE):
    # FC NN DEFINITION
    input_1 = Input(shape=(VECTOR_SIZE,))
    input_2 = Input(shape=(VECTOR_SIZE,))


    def cosine_distance(vecs, normalize=False):
        x, y = vecs
        if normalize:
            x = K.l2_normalize(x, axis=0)
            y = K.l2_normalize(x, axis=0)
        return K.prod(K.stack([x, y], axis=1), axis=1)

    def cosine_distance_output_shape(shapes):
        return shapes[0]

    merged = Lambda(cosine_distance,
                  output_shape=cosine_distance_output_shape)([input_1, input_2])

    #fc1 = Dense(512, kernel_initializer="glorot_uniform")(merged)
    #fc1 = Activation("relu")(fc1)

    fc2 = Dense(128, kernel_initializer="glorot_uniform")(merged)
    fc2 = Activation("relu")(fc2)

    pred = Dense(1, kernel_initializer="glorot_uniform")(fc2)
    pred = Activation("sigmoid")(pred)

    model = Model(inputs=[input_1, input_2], outputs=pred)

    return model

# %% [code]
def nasnet_to_vec(image_dir, vectors_file, sample_per_class):
    IMG_SIZE = 331
    feat_extr = nasnet.NASNetLarge(weights="imagenet", include_top=True)
    #feat_extr.summary()

    model = Model(inputs=feat_extr.input, outputs=feat_extr.layers[-2].output)
    preprocessor = nasnet.preprocess_input

    vectorize_images(image_dir, IMG_SIZE, preprocessor, model, vectors_file, sample_per_class)

def mobilnet_to_vec(image_dir, vectors_file, sample_per_class):
    IMG_SIZE = 224

    feat_extr = mobilenet_v2.MobileNetV2(weights="imagenet", include_top=True)
    #feat_extr.summary()

    model = Model(inputs=feat_extr.input, outputs=feat_extr.layers[-2].output)
    preprocessor = mobilenet_v2.preprocess_input

    vectorize_images(image_dir, IMG_SIZE, preprocessor, model, vectors_file, sample_per_class)

def vgg16_to_vec(image_dir, vectors_file, sample_per_class):

    # GENERATE VECTORS USING VGG16
    IMG_SIZE = 224
    feat_extr = vgg16.VGG16(weights="imagenet", include_top=True)
    feat_extr.summary()

    out = feat_extr.get_layer("block2_conv2").output
    out = GlobalMaxPooling2D()(out)

    model = Model(inputs=feat_extr.input, outputs=out)

    preprocessor = vgg16.preprocess_input

    vectorize_images(image_dir, IMG_SIZE, preprocessor, model, vectors_file, sample_per_class)


def train_model(model, vectors_file, VECTOR_SIZE):
    # SETUP
    image_triples = get_triples(vectors_file)
    train_triples, val_triples, test_triples = train_test_split(image_triples, splits=[0.7, 0.1, 0.2])
    print(len(train_triples), len(val_triples), len(test_triples))

    # VECTORS LOAD
    vec_dict = load_vectors(vectors_file)
    print("vec load ok...")

    # GENERATORS
    train_gen = data_generator(train_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
    val_gen = data_generator(val_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
    print("gen ok...")


    # COMPILE AND TRAIN
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    print("compile ok...")
    train_steps_per_epoch = len(train_triples) // BATCH_SIZE
    val_steps_per_epoch = len(val_triples) // BATCH_SIZE
    history = model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch,
                                  epochs=NUM_EPOCHS,
                                  validation_data=val_gen, validation_steps=val_steps_per_epoch)


def evaluate_model_sigmoid(model, VECTOR_SIZE, vectors_file):
    image_triples = get_triples(vectors_file)

    vec_dict = load_vectors(vectors_file)
    print("vec load ok...")
    train_triples, val_triples, test_triples = train_test_split(image_triples, splits=[0.25, 0.25, 0.5])

    num_test_steps = len(test_triples) // BATCH_SIZE

    test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)

    model.evaluate(test_gen, verbose=1, steps=num_test_steps)



# %% [code]
def select_photos_from_INRIA(path, n_shot, n_way):
    labels = os.listdir(path)
    counter = 0
    for label in labels:
        num_images = len(os.listdir(os.path.join(path, label)))
        if num_images > 20:
            for i, image in enumerate(os.listdir(os.path.join(path, label))[:20]):
                old_label, name = image.split("_")
                new_name = str(int(old_label) + 105) + "_" + name
                if i < 20-n_shot:
                     shutil.copyfile(os.path.join(path, label, image), os.path.join("./data_eval/", new_name))
                else:
                    shutil.copyfile(os.path.join(path, label, image), os.path.join("./merged/", new_name))
        counter += 1
        if counter == n_way:
            break

# %% [code]
'''Questo esperimento intende verificare la capacità di una rete siamese nel distinguere se due foto
rappresentano lo stesso oggetto (in questo caso un fiore).
L'esperimento si realizza come segue:
1) Vengono copiati localmente 2 dataset Oxford 102 e il dataset dei fiori INRIA
2) Vengono fusi utilizzando questo criterio: tutte le foto di oxford 102 e le classi di INRIA con almeno 20 campioni
   di cui 5 verranno utilizzate per fare il training e 15 per la validazione
3) Vengono estratte le features delle foto (con una rete a scelta)
4) Viene allenato il classificatore binario che utilizza come strategia di unificazione la distanza cosenica
'''

!rm -rf "./merged"

!cp -r "/kaggle/input/oxford102" oxford102
!cp -r "/kaggle/input/inria-flowers" inria
!cp -r "./oxford102/flowers/" "./merged/"

!rm -rf data_eval
!mkdir data_eval

select_photos_from_INRIA("./inria/INRIA/images/", 5, 5)

# %% [code]
!ls -l data_eval | wc -l

# %% [code]
'''Questo valuta la capacità della rete di distinguere se due immagini rappresentano lo stesso oggetto.'''

BATCH_SIZE = 32
NUM_EPOCHS = 1

LEARNING_RATE = 0.01
FEATURE_VECTOR_SIZE = 1280

model = get_model(FEATURE_VECTOR_SIZE)


mobilnet_to_vec("./merged/", "merged_5_shot.tsv", 1000)
mobilnet_to_vec("./data_eval/", "merged_eval.tsv", 100)


train_model(model, "merged_5_shot.tsv", FEATURE_VECTOR_SIZE)
evaluate_model_sigmoid(model, FEATURE_VECTOR_SIZE, "merged_eval.tsv")
               
