# -*- coding: utf-8 -*-
import sys

import numpy.linalg as la


def compare(face_descriptor1, face_descriptor2):
    return la.norm(face_descriptor1 - face_descriptor2, axis=0)


def similar_faces(face_descriptors, target_face_descriptor, threshold=0.6):
    faces = []
    for i, fd in enumerate(face_descriptors):
        distance = compare(fd, target_face_descriptor)
        if distance <= threshold:
            faces.append((i, distance))
    return faces


def closest_one(face_descriptors, target_face_descriptor):
    min_idx = 0
    min_distance = sys.maxsize
    for i, fd in enumerate(face_descriptors):
        distance = compare(fd, target_face_descriptor)
        if distance < min_distance:
            min_distance = distance
            min_idx = i
    return min_idx, min_distance
