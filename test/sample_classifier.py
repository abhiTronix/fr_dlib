import os
import sys

import cv2
from img_utils.files import filename, images_in_dir
from img_utils.images import put_text

from fr import compute_face_descriptor, detect_faces, load_samples_descriptors, save2pickle, load_from_pickle
from fr.face_classifier import SVMClassifier


def _labeled(class_names):
    names = list(set(class_names))
    names = sorted(names)
    name_dict = dict((names[i], i) for i in range(len(names)))
    return [name_dict[n] for n in class_names], names


def main(samples_dir, test_images_dir, output_dir):
    # face_descriptors, class_names = load_samples_descriptors(samples_dir)
    # save2pickle(face_descriptors, class_names, 'wg_merged.pkl')
    face_descriptors, class_names = load_from_pickle('wg_merged.pkl')
    print(class_names)
    labels, names = _labeled(class_names)
    classifier = SVMClassifier()

    print([names[i] for i in labels])

    classifier.train(face_descriptors, labels, names)

    # classifier.load("classifier_2018-05-15 13:30:06.213832.pkl")
    image_files = images_in_dir(test_images_dir)
    for im_f in image_files:
        output_path = os.path.join(output_dir, filename(im_f))
        im = cv2.imread(im_f)
        faces = detect_faces(im)
        for face in faces:
            descriptor = compute_face_descriptor(im, face)
            results = classifier.predict([descriptor])
            for r in results:
                txt = '{}'.format(r)
                put_text(im, txt, font_face=cv2.FONT_HERSHEY_SIMPLEX)

                print('{}: {} '.format(im_f, r))
        cv2.imwrite(output_path, im)


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 4:
        print('---------------------------------------------------------------------------')
        print('|-- Usage: python sample_classifier.py ${samples_dir} ${test_data_dir} ${output_dir}')
        print('---------------------------------------------------------------------------')
        exit(0)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
