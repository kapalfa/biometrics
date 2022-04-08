import os
import cv2
from sklearn.metrics import classification_report

M_COEFF = 0.75

base = {}
suspects = {}

directory = 'DB1_B'
for filename in os.listdir("DB1_B"):
    path = "DB1_B" + '/' + filename

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, orb_des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    # img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    # plt.imshow(img2), plt.show()

    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    # img=cv2.drawKeypoints(gray,kp,img)
    # cv2.imwrite('sift_keypoints.jpg',img)

    # img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('sift_keypoints.jpg',img)
    kp, sift_des = sift.detectAndCompute(gray, None)

    feat_dict = {'sift': sift_des, 'orb': orb_des}

    img_id = filename.split('.')[0]
    if img_id.split('_')[-1] == '1':
        base[img_id.split('_')[0]] = feat_dict
    else:
        suspects[img_id] = feat_dict

    ##part 2
orb_bf = cv2.BFMatcher(cv2.NORM_HAMMING)
sift_bf = cv2.BFMatcher(cv2.NORM_L2)

orb_test = []
orb_pred = []

sift_test = []
sift_pred = []

for key, value in suspects.items():

    orb_des = value['orb']
    sift_des = value['sift']

    best_matches_orb = None
    best_matches_sift = None
    best_matches_dict = {}
    good_match_num = 0
    good_match_num2 = 0
    max = -1
    max2 = -1

    for key2, value2 in base.items():
        orb2_des = value2['orb']

        sift2_des = value2['sift']
        sift_matches = sift_bf.knnMatch(sift_des, sift2_des, k=2)
        orb_matches = orb_bf.knnMatch(orb_des, orb2_des, k=2)


        good_match_num = 0
        for m, n in orb_matches:
            if m.distance < M_COEFF * n.distance:
                good_match_num += 1
            if good_match_num > max:
                max = good_match_num
                best_matches_orb = key2

        good_match_num2 = 0
        for m, n in sift_matches:
            if m.distance < M_COEFF * n.distance:
                good_match_num2 += 1
            if good_match_num2 > max2:
                max2 = good_match_num2
                best_matches_sift = key2

    orb_pred.append(best_matches_orb)
    orb_test.append(key.split('_')[0])
    sift_pred.append(best_matches_sift)
    sift_test.append(key.split('_')[0])



#orb prints
print(classification_report(orb_test, orb_pred))

#sift prints
print(classification_report(sift_test, sift_pred))