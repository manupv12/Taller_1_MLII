import numpy as np
import gdown
import os
import glob
import cv2

class Pictures:
    def __init__(self, path_folder=None, picture_name=None, link_drive=None):
        self.path_folder = path_folder
        self.picture_name = picture_name
        self.path_picture = os.path.join(path_folder, picture_name)
        self.link_drive = link_drive
    def tf_folder(self):
        isExist = os.path.exists(self.path_folder)
        return isExist
    def tf_picture(self):
        isExist = os.path.exists(self.path_picture)
        return isExist
    def download_files(self):
        try:
            gdown.download_folder(self.link_drive, quiet=True, use_cookies=False)
            return "succes"
        except:
            return "fail"
        
    def load_pictures(self):
        paths_images = glob.glob(os.path.join(self.path_folder, "*"))
        images = []
        self_picture = 0
        for path_img in paths_images:
            try:
                image = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (256, 256))
                images.append(image)
                if path_img == self.path_picture:
                    self_picture = image
            except:
                pass
        self.images = images
        self.self_picture = self_picture
        self.self_picture_path = os.path.join(self.path_folder, "self_picture.jpeg")
        cv2.imwrite(os.path.join(self.path_folder, "self_picture.jpeg"), self_picture)
        return images, self_picture
    def mean_picture(self):
        mean = np.mean(np.array(self.images), axis=0)
        cv2.imwrite(os.path.join(self.path_folder, "mean.jpeg"), mean)
        self.mean = mean
        self.mean_path = os.path.join(self.path_folder, "mean.jpeg")
        return mean
    def distance(self):
        mean = self.mean
        self_picture = self.self_picture
        dist = np.linalg.norm(mean.flatten() - self_picture.flatten())
        return dist
