import glob
import os
import sys
import time

import cv2
from img_utils.files import images_in_dir

from fr import detect_faces, compute_face_descriptor, load_samples_descriptors, read_from_hierarchy
from fr.face_classifier import SVMClassifier

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


def main(samples_dir, validate_data_dir, output_dir):
    face_descriptors, class_names = load_samples_descriptors(samples_dir)
    print("len face_descriptors: {}".format(len(face_descriptors)))
    print("len class_names: {}".format(len(class_names)))
    print("class_names: {}".format(class_names))
    print("class_names: {}".format(class_names))
    print("total class: {}".format(len(set(class_names))))

    labels, names = _labeled(class_names)
    classifier = SVMClassifier(probability=True)

    print([names[i] for i in labels])

    classifier.train(face_descriptors, labels, names)

    hierarchy = read_from_hierarchy(validate_data_dir)

    total = 0
    accurate = 0
    for k in hierarchy.keys():
        image_files = images_in_dir(hierarchy[k])
        for im_f in image_files:
            total += 1
            im = cv2.imread(im_f)
            faces = detect_faces(im)
            start = time.time()
            for face in faces:
                descriptor = compute_face_descriptor(im, face)
                results = classifier.predict([descriptor])
                # idx, distance = closest_one(face_descriptors, descriptor)
                # if distance > 0.4:
                #     label = 'unknown'
                # else:
                #     label = class_names[idx]
                label, probability = results[0]
                if probability < 0.5:
                    label = 'unknown'
                if label == k:
                    accurate += 1

                subdir = _subdir(output_dir, label)

                images_count = len(glob.glob('{}/*.jpg'.format(subdir)))

                f_name = '{}_{}.jpg'.format(label, '{0:04d}'.format(images_count))

                # print('{}: {}, of distance :{} '.format(im_f, f_name, distance))

                output_path = os.path.join(subdir, f_name)
                cv2.imwrite(output_path, im)
            print('{} done, time spent: {} '.format(im_f, time.time() - start))

        print("Accuracy is: {} ".format(accurate / total))

    print('Accurate is {}, total: {}'.format(accurate, total))
    print("Total Accuracy is: {} ".format(accurate / total))


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 4:
        print('---------------------------------------------------------------------------')
        print('|-- Usage: python sample_clustering.py ${samples_dir} ${validate_data_dir} ${output_dir}')
        print('---------------------------------------------------------------------------')
        exit(0)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
