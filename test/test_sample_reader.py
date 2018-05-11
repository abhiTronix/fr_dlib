import os

import numpy as np

from fr import load_samples_descriptors, save2pickle, load_from_pickle

BASE_DIR = os.path.dirname(__file__)
images_dir = os.path.join(BASE_DIR, "images")


def test_load_samples_descriptors():
    face_descriptors, class_names = load_samples_descriptors(images_dir)
    assert len(face_descriptors) == 3
    assert len(class_names) == 3
    assert class_names[0] == "Aaron_Eckhart"
    assert class_names[1] == "Blythe_Hartley"
    assert class_names[2] == "Blythe_Hartley"


def test_save_and_load():
    face_descriptors, class_names = load_samples_descriptors(images_dir)
    print(face_descriptors)
    print(class_names)
    save2pickle(face_descriptors, class_names, os.path.join(BASE_DIR, "descriptors.pkl"))
    fd, names = load_from_pickle("descriptors.pkl")
    print(fd)
    print(names)
    assert np.array_equal(fd, face_descriptors) is True
    assert names == class_names


# test_save_and_load()
