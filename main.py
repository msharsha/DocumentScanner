# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:28:53 2019

@author: Manoj Sree Harsha
"""

from transform import four_point_transform
import numpy as np
import cv2

image = cv2.imread("sample.jpg")
#region in which you want to perform tranformation
pts = [(73, 239), (356, 117), (475, 265), (187, 443)]
warped = four_point_transform(image, pts)
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
