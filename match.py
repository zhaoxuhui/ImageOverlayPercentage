# coding=utf-8
import cv2
import numpy as np
import os


def findAllFiles(root_dir, filter):
    print("Finding files ends with \'" + filter + "\' ...")
    separator = os.path.sep
    paths = []
    names = []
    files = []
    # 遍历
    for parent, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filter):
                paths.append(parent + separator)
                names.append(filename)
    for i in range(paths.__len__()):
        files.append(paths[i] + names[i])
    print (names.__len__().__str__() + " files have been found.")
    paths.sort()
    names.sort()
    files.sort()
    return paths, names, files


def getSURFkps(img):
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=2000)
    kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
    return kp, des


def matchSURFkps(kp1, des1, kp2, des2):
    good_matches = []
    good_kps1 = []
    good_kps2 = []

    good_out = []
    good_out_kp1 = []
    good_out_kp2 = []

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 筛选
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append(matches[i])
            good_kps1.append(kp1[matches[i][0].queryIdx])
            good_kps2.append(kp2[matches[i][0].trainIdx])

    print("good matches:" + good_matches.__len__().__str__())
    for i in range(good_kps1.__len__()):
        good_out_kp1.append([good_kps1[i].pt[0], good_kps1[i].pt[1]])
        good_out_kp2.append([good_kps2[i].pt[0], good_kps2[i].pt[1]])
        good_out.append([good_kps1[i].pt[0], good_kps1[i].pt[1], good_kps2[i].pt[0], good_kps2[i].pt[1]])
    return good_out_kp1, good_out_kp2, good_out


if __name__ == '__main__':
    _, names, img_files = findAllFiles(".", ".jpg")

    img_w = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE).shape[1]
    img_h = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE).shape[0]
    total_pixel = img_w * img_h

    kp1s = []
    kp2s = []
    matches = []
    nums = []
    blank_percents = []
    overlay_percents = []

    # 新建文本文件用于记录结果
    txt_file = open("overlay_records.txt", "a+")
    txt_file.write(
        "Kp1\tKp2\tMatches\tNums\tTotal\tBlank Percent\tOverlay Percent\n")
    txt_file.close()

    for i in range(img_files.__len__() - 1):
        print i + 2, "/", img_files.__len__()
        # 读取影像
        img1 = cv2.imread(img_files[i])
        img2 = cv2.imread(img_files[i + 1])
        # 提取特征点
        kp1, des1 = getSURFkps(img1)
        kp2, des2 = getSURFkps(img2)
        kp1s.append(kp1.__len__())
        kp2s.append(kp2.__len__())
        print "kp1:", kp1.__len__(), "kp2:", kp2.__len__()
        # 特征匹配
        g1, g2, match = matchSURFkps(kp1, des1, kp2, des2)
        matches.append(match.__len__())
        # 计算单应矩阵
        h, res = cv2.findHomography(np.array(g2), np.array(g1))
        print h
        # 基于单应矩阵重采
        dst = cv2.warpPerspective(img2, h, (img1.shape[1], img1.shape[0]))
        # 统计灰度为0像素个数
        num = np.sum(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) == 0)
        nums.append(num)
        # 计算百分比
        blank_percent = (num * 1.0 / total_pixel) * 100
        over_percent = 100 - (num * 1.0 / total_pixel) * 100
        blank_percents.append(blank_percent)
        overlay_percents.append(over_percent)
        print "blank pixel num:", num
        print "total pixel num:", total_pixel
        print "blank percent:", blank_percent, "%"
        print "overlay percent:", over_percent, "%"
        print "-" * 50
        # 写入文件
        txt_file = open("overlay_records.txt", "a+")
        txt_file.write(
            kp1s[i].__str__() + "\t" + kp2s[i].__str__() + "\t" +
            matches[i].__str__() + "\t" +
            nums[i].__str__() + "\t" +
            total_pixel.__str__() + "\t" +
            blank_percents[i].__str__() + "\t" +
            overlay_percents[i].__str__() + "\n")
        txt_file.close()
