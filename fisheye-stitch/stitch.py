# -*- coding:utf-8 -*-
# by yifei 2020.11

import glob
import os
import sys
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_K_and_D(checkerboard, imgsPath):
    # global gray
    CHECKERBOARD = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW # +cv2.fisheye.CALIB_CHECK_COND
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints, imgpoints = [], []
    images = sorted(os.listdir(imgsPath))
    images = images[0:len(images):9]
    # images = glob.glob(imgsPath + '/*.jpg')  # source .png
    for fname in images:
        img = cv2.imread(os.path.join(imgsPath,fname))
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        cv2.drawChessboardCorners(img,CHECKERBOARD,corners,ret)
        # cv2.imshow("detect front points",cv2.resize(img,(960,510)))
        # cv2.waitKey(10)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            cv2.drawChessboardCorners(img,CHECKERBOARD,corners,ret)
            cv2.waitKey(10)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    DIM = _img_shape[::-1]
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    return DIM, K, D

def make_coord(shape=(720, 1280)):
    x = np.arange(0, shape[1]).reshape(1, -1)+0.5
    x = np.tile(x, (shape[0], 1))
    y = np.arange(0, shape[0]).reshape(-1, 1)+0.5
    y = np.tile(y, (1, shape[1]))

    coord = np.zeros((*shape, 3), dtype=np.float32)
    coord[:,:,0] = x
    coord[:,:,1] = y
    return coord

def undistort(img_path,d,K,D,DIM,scale=0.6,imshow=False):
    # img = cv2.imread(img_path)
    img = make_coord()
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if dim1[0]!=DIM[0]:
        img = cv2.resize(img,DIM,interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    if scale:#change fov
        Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_32FC1)
    map3 = np.stack((map1,map2),axis=2)
    print(map1.shape,map2.shape,map3.shape)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # np.save(root+"/%s_test.npy"%d,undistorted_img)
    undis_img = scale_img(undistorted_img,d)
    if imshow:
        cv2.imshow("undistorted", undis_img)
        cv2.waitKey(1000)
    return undis_img

class calib_info(object):
    def __init__(self,txt_path):
        self.txt_path = txt_path
        self.paras = self.read_txt()

    def read_txt(self):
        with open(self.txt_path,'r') as info:
            para = info.readlines()
            return para

    def get_front(self):
        p = self.paras[0].split('*')
        matrix = []
        for i in range(len(p)):
            DIM_K_D = np.array(eval(p[i].split('=')[-1]))
            matrix.append(DIM_K_D)
        return matrix[0],matrix[1],matrix[2],matrix[3]

    def get_rear(self):
        p = self.paras[1].split('*')
        matrix = []
        for i in range(len(p)):
            DIM_K_D = np.array(eval(p[i].split('=')[-1]))
            matrix.append(DIM_K_D)
        return matrix[0],matrix[1],matrix[2],matrix[3]

    def get_left(self):
        p = self.paras[2].split('*')
        matrix = []
        for i in range(len(p)):
            DIM_K_D = np.array(eval(p[i].split('=')[-1]))
            matrix.append(DIM_K_D)
        return matrix[0],matrix[1],matrix[2],matrix[3]

    def get_right(self):
        p = self.paras[3].split('*')
        matrix = []
        for i in range(len(p)):
            DIM_K_D = np.array(eval(p[i].split('=')[-1]))
            matrix.append(DIM_K_D)
        return matrix[0],matrix[1],matrix[2],matrix[3]

def save_calib(infos,direction):
    DIM, K, D = infos
    l1 = "DIM=" + str(_img_shape[::-1])
    l2 = "K=np.array(" + str(K.tolist()) + ")"
    l3 = "D=np.array(" + str(D.tolist()) + ")"
    path = root + "/calib_cam_4.txt"
    with open(path,'a+') as f:
        f.writelines("%s:%s*%s*%s\n"%(direction,l1,l2,l3))

def map_bev(img,M,d_size):
    size = (1440,720)
    warped = cv2.warpPerspective(img, M, size)
    return warped

def rotate_trans(image, angle, d):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the angle to rotate clockwise), then grab the sine and cosine
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    rotate_mat = M
    image = cv2.warpAffine(image, rotate_mat, (nW, nH))
    # cv2.imshow("img",image)
    # cv2.waitKey(3000)
    front_trans_mat = np.float32([[1,0,290],[0,1,136]]) # front 
    right_trans_mat = np.float32([[1,0,1035],[0,1,210]]) # right
    left_trans_mat = np.float32([[1,0,255],[0,1,203]]) # left
    rear_trans_mat = np.float32([[1,0,265],[0,1,1076]]) # rear
    translation = {"front":front_trans_mat,"right":right_trans_mat,"left":left_trans_mat,"rear":rear_trans_mat}
    # img = cv2.warpAffine(image, translation[d], (2000,2000))
    # cv2.imwrite("warped_%s.jpg"%d,img)
    return cv2.warpAffine(image, translation[d], (2000,2000))[500:1500,500:1500]

def desplay(img):
    cv2.imshow("1",img)
    cv2.waitKey(1000)

def cal_num(img_path):
    nums = []
    for i in img_path:
        num = len(os.listdir(root+'/'+i[1:]))
        nums.append(num)
    return min(nums)

def cal_point(img,h,w):
    neighbors = [(1, 1), (1, -1), (1, 0), (-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1)]
    sums = 0
    for n in neighbors:
        dr, dc = n
        try:
            if sum(img[h+dr][w+dc]) == 0:
                sums += 1
        except IndexError:
            pass
    if sums >= 3:
        img[h,w] = [0,0,0]
    return img

def scale_img(img,d):
    h, w, _ = img.shape
    if d in ["front","rear"]:
        scale_img = cv2.resize(img,(int(w*3/2),int(h*3/2))) # 1920*1080
        rear_undis = scale_img[int(h/4):int(h/4)+h,int(w/4):int(w/4)+w] # 1280*720 
        return rear_undis
    else:
        return img

def cal_lap(mats): # 0:front 1:right 2:rear 3:lefts
    mask1 = (mats[0] & mats[1])
    mask2 = (mats[1] & mats[2])
    mask3 = (mats[2] & mats[3])
    mask4 = (mats[0] & mats[3])
    return [mask1,mask2,mask3,mask4]

def cal_ovelap(mats):
    mask1 = (mats[0] & mats[1])
    mask2 = (mats[1] & mats[2])
    mask3 = (mats[2] & mats[3])
    mask4 = (mats[0] & mats[3])
    # return [(mask1 | mask4),(mask1 | mask2),(mask2 | mask3),(mask3 | mask4)]
    return [((mask1 | mask4) ^ mats[0]),((mask1 | mask2) ^ mats[1]),((mask2 | mask3) ^ mats[2]),((mask3 | mask4) ^ mats[3])]
    #  not (front overlap, right overlap, rear overlap, left overlap)

def process(img):
    h,w,_ = img.shape
    pro_img = img.copy()
    for i in range(int(h/2),h):
        for j in range(int(w/2),w):
            process_img = cal_point(pro_img,i,j,)
    return process_img

class cal_weight(object):

    def __init__(self,mask,img,imgs):
        self.mask = mask
        self.img0 = img
        self.img = img.copy()
        self.loc0 = zip(np.where(mask[0])[0],np.where(mask[0])[1])
        self.loc1 = zip(np.where(mask[1])[0],np.where(mask[1])[1])
        self.loc2 = zip(np.where(mask[2])[0],np.where(mask[2])[1])
        self.loc3 = zip(np.where(mask[3])[0],np.where(mask[3])[1]) 
        self.front_left_conrner = (np.where(mask[3])[0].max(),np.where(mask[3])[1].max())
        self.front_right_conrner = (np.where(mask[0])[0].max(),np.where(mask[0])[1].min())
        self.left_rear_conrner = (np.where(mask[2])[0].min(),np.where(mask[2])[1].max())
        self.right_rear_conrner = (np.where(mask[1])[0].min(),np.where(mask[1])[1].min())
        self.front_img = imgs[0]
        self.right_img = imgs[1]
        self.rear_img = imgs[2]
        self.left_img = imgs[3]

    def f_r_angle(self):
        end1 = np.ones((1000,1000,2),dtype=np.float32)
        end2 = np.ones((1000,1000,2),dtype=np.float32)
        print(end1.shape)
        w = []
        for i in self.loc0:
            weight = np.arctan((abs(self.front_right_conrner[0] - i[0])+0.001)/(abs(i[1]-self.front_right_conrner[1])+0.001))
            weight = weight/(np.pi/2)
            if weight < 0.276:
                weight = 0
            elif weight > 0.813:
                weight =1
            else:
                weight = (weight-0.276) * (1/(0.813-0.276))
            self.img[i] = self.front_img[i]*weight + self.right_img[i]*(1-weight)
            w.append(weight)
            # end1[i[0],i[1]] = weight
            # end2[i[0],i[1]] = 1-weight
        # np.save("./test/front_weight_r.npy",end1)
        # np.save("./test/right_weight_f.npy",end2)
        print('front_right',max(w),min(w))
        return self.img

    def r_r_angle(self):
        end1 = np.ones((1000,1000,2),dtype=np.float32)
        end2 = np.ones((1000,1000,2),dtype=np.float32)
        w = []
        for i in self.loc1:
            weight = np.arctan((abs(self.right_rear_conrner[0] - i[0])+0.001)/(abs(i[1]-self.right_rear_conrner[1])+0.001))
            weight = weight/(np.pi/2)
            if weight < 0.350: # min boundary
                weight = 0
            elif weight > 0.811: # max boundary
                weight = 1
            else:
                weight = (weight-0.350) * (1/(0.811-0.350))
            self.img[i] = self.rear_img[i]*weight + self.right_img[i]*(1-weight)
            w.append(weight)
            # end1[i[0],i[1]] = weight
            # end2[i[0],i[1]] = 1-weight
        # np.save("./test/right_weight_r.npy",end2)
        # np.save("./test/rear_weight_r.npy",end1)
        print('right_rear',max(w),min(w))
        return self.img

    def r_l_angle(self):
        end1 = np.ones((1000,1000,2),dtype=np.float32)
        end2 = np.ones((1000,1000,2),dtype=np.float32)
        w = []
        for i in self.loc2:
            weight = np.arctan((abs(self.left_rear_conrner[0] - i[0])+0.001)/(abs(i[1]-self.left_rear_conrner[1])+0.001))
            weight = weight/(np.pi/2)
            if weight > 0.793:
                weight = 1
            elif weight < 0.228:
                weight = 0
            else:
                weight = (weight-0.228) * (1/(0.793-0.228))
            self.img[i] = self.rear_img[i]*weight + self.left_img[i]*(1-weight)
            w.append(weight)
            # end1[i[0],i[1]] = weight
            # end2[i[0],i[1]] = 1-weight
        # np.save("./test/rear_weight_l.npy",end1)
        # np.save("./test/left_weight_r.npy",end2)
        print('right_left',max(w),min(w))
        return self.img
    
    def l_f_angle(self):
        end1 = np.ones((1000,1000,2),dtype=np.float32)
        end2 = np.ones((1000,1000,2),dtype=np.float32)
        w = []
        for i in self.loc3:
            weight = np.arctan((abs(self.front_left_conrner[0] - i[0])+0.001)/(abs(i[1]-self.front_left_conrner[1])+0.001))
            weight = weight/(np.pi/2)
            if weight > 0.838:
                weight = 1
            elif weight < 0.272:
                weight = 0
            else:
                weight = (weight-0.272) * (1/(0.838-0.272))
            self.img[i] = self.front_img[i]*weight + self.left_img[i]*(1-weight)
            w.append(weight)
            # end1[i[0],i[1]] = weight
            # end2[i[0],i[1]] = 1-weight
        # np.save("./test/left_weight_f.npy",end2)
        # np.save("./test/front_weight_l.npy",end1)
        print('left_front',max(w),min(w))
        return self.img

def stitch_imgs(images,save_sign=False):
    l = np.zeros((1000,1000,3),dtype="uint8")
    l_mask = np.zeros((1000,1000,1),dtype="uint8")
    masks,imgs = [],[]
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > 5
        masks.append(mask)
        imgs.append(img)

    if save_sign:
        # record overlap region or not 
        # 0:invalid 1:front&right 2:right&rear 3:rear&left 4:left&front 5:front 6:right 7:rear 8:left
        no_overlap = cal_ovelap(masks) # depend region
        overlap = cal_lap(masks) # overlap region
        for j in range(4):
            front_right = overlap[j]
            loc = np.where(front_right)
            locs = np.stack((loc[0],loc[1]),axis=1)
            front = no_overlap[j]
            loc1 = np.where(front)
            locs1 = np.stack((loc1[0],loc1[1]),axis=1)
            print(locs.shape,locs1.shape)
            lens, _ = locs.shape
            lens1,_ = locs1.shape
            if j < 3:
                num = j+6
            elif j == 3:
                num = j+2
            for i in range(lens):
                i0,j0 = locs[i]
                l_mask[i0,j0] += num
            for k in range(lens1):
                i1,j1 = locs1[k]
                l_mask[i1,j1] += j+1
        np.save("end_mask.npy",l_mask)
    
    non_overlap_mtx = cal_ovelap(masks)
    for j in range(len(masks)):
        m = non_overlap_mtx[j]
        l[m] = imgs[j][m]
    overlap_mtx = cal_lap(masks)

    # for k in range(len(imgs)):
    #     if k == 3:
    #         l[overlap_mtx[k]] = imgs[k][overlap_mtx[k]]*0.5 + imgs[k-3][overlap_mtx[k]]*0.5
    #     else:
    #         l[overlap_mtx[k]] = imgs[k][overlap_mtx[k]]*0.5 + imgs[k+1][overlap_mtx[k]]*0.5
    # for k in range(len(imgs)):
    img_s = cal_weight(overlap_mtx,l,imgs).f_r_angle()
    img_s = cal_weight(overlap_mtx,img_s,imgs).r_r_angle()
    img_s = cal_weight(overlap_mtx,img_s,imgs).r_l_angle()
    img_s = cal_weight(overlap_mtx,img_s,imgs).l_f_angle()
    # cv2.imshow("m",img_s)
    # cv2.waitKey(5000)
    # cv2.imwrite(root+"/end_1130.jpg",img_s)
    return img_s

class Color_calib(object):

    def __init__(self,images,overlap):
        self.front_img = images[0]/255.0
        self.right_img = images[1]/255.0
        self.rear_img = images[2]/255.0
        self.left_img = images[3]/255.0
        self.front_right_map = overlap[0]
        self.right_rear_map = overlap[1]
        self.rear_left_map = overlap[2]
        self.left_front_map = overlap[3]
        self.Af = self.make_rgb(self.front_img,self.left_front_map)
        self.Al = self.make_rgb(self.left_img,self.left_front_map)
        self.Bf = self.make_rgb(self.front_img,self.front_right_map)
        self.Br = self.make_rgb(self.right_img,self.front_right_map)
        self.Cl = self.make_rgb(self.left_img,self.rear_left_map)
        self.Ct = self.make_rgb(self.rear_img,self.rear_left_map)
        self.Dr = self.make_rgb(self.right_img,self.right_rear_map)
        self.Dt = self.make_rgb(self.rear_img,self.right_rear_map)
        self.r_para = self.adjust_rgb(self.mat_rgb()[2])
        self.g_para = self.adjust_rgb(self.mat_rgb()[1])
        self.b_para = self.adjust_rgb(self.mat_rgb()[0])
        print(self.r_para,self.g_para,self.b_para)
    
    @staticmethod
    def make_rgb(img,mtx):
        img0 = img.copy()
        img0[~mtx] = 0
        # cv2.imshow("img",img0)
        # cv2.waitKey(2000)
        # print(";;;;")
        r = img[:,:,0][mtx].mean()
        g = img[:,:,1][mtx].mean()
        b = img[:,:,2][mtx].mean()
        return [r,g,b]


    def mat_rgb(self):
        mtxs = []
        for i in range(3):
            mat = np.array([[self.Af[i]**2+self.Bf[i]**2, -self.Af[i]*self.Al[i], -self.Bf[i]*self.Br[i], 0],
            [-self.Af[i]*self.Al[i], self.Al[i]**2+self.Cl[i]**2, 0, -self.Cl[i]*self.Ct[i]],
            [-self.Bf[i]*self.Br[i], 0, self.Br[i]**2+self.Dr[i]**2, -self.Dr[i]*self.Dt[i]],
            [0, -self.Cl[i]*self.Ct[i], -self.Dr[i]*self.Dt[i], self.Ct[i]**2+self.Dt[i]**2]])
            # print(mat)
            s = np.linalg.solve(mat,np.zeros((4,1))+0.01)
            s = s.flatten()
            s = s/s.max()
            mtxs.append(s)
        # print("llll",s)
        return mtxs

    @staticmethod
    def adjust_rgb(p):
        param = [p[0],p[2],p[3],p[1]]
        return param

if __name__ == '__main__':
    global root
    # root = os.path.dirname(__file__)
    root = os.getcwd()
    mode = "undis"
    cam_a4 = {"1right":90,"0front":0,"2rear":180,"3left":-90}
    stitch = False
    surround_img = np.zeros((1000,1000,3),dtype=np.ubyte)
    cam_4 = cam_a4.keys()
    if mode == "calib":
        check_size = (8,6) # calib checkboard size
        for d in cam_4:
            img_path = root + '/' + d
            DIM, K, D = get_K_and_D(check_size,img_path)
            save_calib((DIM,K,D),d)
    elif mode == "undis":
        calib_path = root+'/calib_para.txt'
        calib = calib_info(calib_path)
        function = {"right":calib.get_right(),"front":calib.get_front(),"rear":calib.get_rear(),"left":calib.get_left()}
        imgs, show = [], []
        for d0 in sorted(cam_4):
            d = d0[1:]
            DIM, K, D, M = function.get(d)
            DIM = tuple(DIM.tolist())
            img_path = root + '/' + d + '.jpg'
            imgs_0 = cv2.imread(img_path) # 1280*720
            start = time.time()
            undis_img = undistort(img_path,d,K,D,DIM) # 1280*720
            warped_img = map_bev(undis_img,M,d) # 1440*720
            # np.save("./warp_%s_test.npy"%d,warped_img)
            show_img = np.hstack((imgs_0,undis_img))
            show_img = cv2.resize(show_img,(1280,360))
            show.append(show_img)
            rotate_img = rotate_trans(warped_img,cam_a4[d0],d)
            # cv2.imshow("rota_img",rotate_img)
            # cv2.waitKey(3000)
            # np.save("./test/warp_rt_%s.npy"%d,rotate_img)
            imgs.append(rotate_img)
            mask = rotate_img > 0
            surround_img[mask] = rotate_img[mask]
            print('stage2',time.time()-start,'ms')
        # cv2.imwrite("sur.jpg",surround_img)
        if stitch:
            write_img = surround_img
            cv2.imshow("surround img", cv2.resize(write_img,(1000,1000)))
            cv2.waitKey(2000)
        else:
            start_end = time.time()
            su_img = stitch_imgs(imgs)
            print('stage2',time.time()-start_end,'ms')
            all = np.hstack((np.vstack((show[0],show[1],show[2],show[3])),cv2.resize(su_img,(1440,1440))))
            end_img = cv2.resize(all,(1360,720))
            # cv2.imwrite("end_1202.jpg",end_img)
            # cv2.imshow("surround img",end_img)
            # if cv2.waitKey() == 27:
            #     cv2.destroyAllWindows()