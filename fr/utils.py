# -*- coding: utf-8 -*-

import dlib


def to_dlib_rectangles(mmod_rectangles):
    rectangles = dlib.rectangles()
    rectangles.extend([d.rect for d in mmod_rectangles])
    return rectangles


def to_rectangle(dlib_rectangle):
    return dlib_rectangle.left(), dlib_rectangle.top(), dlib_rectangle.right(), dlib_rectangle.bottom()


def rect_2_dlib_rectangles(pnt1, pnt2):
    x1, y1 = pnt1
    x2, y2 = pnt2
    rectangles = dlib.rectangles()
    rectangles.extend([dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)])
    return rectangles
