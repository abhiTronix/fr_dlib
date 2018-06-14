import glob
import os
import sys
import time
from collections import Collection, Counter

import cv2
from img_utils.files import images_in_dir

from fr import compute_face_descriptor, read_from_hierarchy, closest_one, load_samples_descriptors, \
    compute_face_descriptor_multi_thread
from fr.face_classifier import SVMClassifier
from fr.utils import rect_2_dlib_rectangles

BASE_DIR = os.path.dirname(__file__)


def _subdir(output_dir, label):
    dir_name = os.path.join(output_dir, label)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    return dir_name


def _labeled(class_names):
    names = list(set(class_names))
    names = sorted(names)
    name_dict = dict((names[i], i) for i in range(len(names)))
    return [name_dict[n] for n in class_names], names


def _save_result(im, label, output_dir):
    subdir = _subdir(output_dir, label)

    images_count = len(glob.glob('{}/*.jpg'.format(subdir)))

    f_name = '{}_{}.jpg'.format(label, '{0:04d}'.format(images_count))

    output_path = os.path.join(subdir, f_name)
    cv2.imwrite(output_path, im)


def _evaluate_with_distance(im, faces, face_descriptors, class_names, label_y, output_dir):
    print('Validate data with label of {}'.format(label_y))
    accurate = 0
    time_total = 0
    for face in faces:
        start = time.time()
        descriptor = compute_face_descriptor(im, face)
        print('Descriptor computing, time spent: {} '.format(time.time() - start))
        idx, distance = closest_one(face_descriptors, descriptor)
        time_spent = time.time() - start
        time_total += time_spent
        print('Distance, time spent: {} '.format(time_spent))
        if distance > 0.4:
            label = 'unknown'
        else:
            label = class_names[idx]
        if label == label_y:
            accurate += 1
        _save_result(im, label, output_dir)
    return accurate, time_total


def _evaluate_with_classifier(im, faces, classifier, label_y, output_dir):
    print('Validate data with label of {}'.format(label_y))
    accurate = 0
    time_total = 0
    for face in faces:
        start = time.time()
        # descriptor = compute_face_descriptor(im, face, upsample=25)
        # descriptors = [descriptor]
        descriptors = compute_face_descriptor_multi_thread(im, face, num_jitters=10, threads=5)
        results = classifier.predict(descriptors)
        time_spent = time.time() - start
        time_total += time_spent
        print('Evaluation, time spent: {} '.format(time_spent))
        # label = results[0]
        label = Counter(results).most_common(1)[0][0]
        if label == label_y:
            accurate += 1
        _save_result(im, label, output_dir)
    return accurate, time_total


def main(samples_dir, validate_data_dir, output_dir):
    classifier_file = "classifier_wgers_160_2018-06-14.pkl"
    classifier = SVMClassifier(probability=False)
    if classifier_file is None:
        face_descriptors, class_names = load_samples_descriptors(samples_dir)
        print("len face_descriptors: {}".format(len(face_descriptors)))
        print("len class_names: {}".format(len(class_names)))
        print("class_names: {}".format(class_names))
        print("class_names: {}".format(class_names))
        print("total class: {}".format(len(set(class_names))))
        labels, names = _labeled(class_names)
        print([names[i] for i in labels])
        classifier.train(face_descriptors, labels, names)
    else:
        classifier.load(classifier_file)

    hierarchy = read_from_hierarchy(validate_data_dir)

    total = 0
    accurate = 0
    time_spent = 0
    for k in hierarchy.keys():
        image_files = images_in_dir(hierarchy[k])
        total += len(image_files)
        for im_f in image_files:
            im = cv2.imread(im_f)

            # faces = detect_faces(im)
            height, width = im.shape[:2]
            faces = rect_2_dlib_rectangles((0, 0), (width, height))

            a, t = _evaluate_with_classifier(im, faces, classifier, k, output_dir)
            accurate += a
            time_spent += t
        print("-------------------------------------Accuracy is: {} ".format(accurate / total))
    print('==================================================================')
    print('Accurate is {}, total: {}'.format(accurate, total))
    print('Total time spent is: {}'.format(time_spent))
    print("Average time spent is: {}".format(time_spent / total))
    print("Average Accuracy is: {} ".format(accurate / total))


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 4:
        print('---------------------------------------------------------------------------')
        print('|-- Usage: python sample_clustering.py ${samples_dir} ${validate_data_dir} ${output_dir}')
        print('---------------------------------------------------------------------------')
        exit(0)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
