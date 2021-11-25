from PIL import Image
from os import listdir
import numpy as np

ignore = ["morralla", ".DS_Store", "contempt"]
all_emotions = ['sadness', 'happiness', 'surprise',
                'neutral', 'fear', 'disgust', 'anger']
paths_default = ["data/all/train/", "data/all/test/"]


class Datos():
    def __init__(self, paths=paths_default, emotions=all_emotions,h=48,w=48):
        self.dim = (h,w)
        self.imgs = []
        self.other_imgs = []
        self.state = []
        self.other_state = []
        self.paths = paths
        if len(emotions) == 7:
            self.emotions = emotions
        elif len(emotions) < 7:
            self.emotions = [x for x in all_emotions if x in emotions]
            self.emotions.append('other')
        else: 
            print("emotions list too large, max 7")

        self.other_emotions = [
            x for x in all_emotions if x not in self.emotions]

    def imgs_load_initial(self):
        '''
        Load the image's directions in the paths given
        '''
        if len(self.imgs):
            print("Images already loaded, plesase clean before load again.")
            return False

        for path in self.paths:
            for item in listdir(path):
                if item not in ignore:
                    if item in self.emotions:
                        self.imgs.extend(
                            [f"{path}{item}/{p}" for p in listdir(f"{path}{item}")])
                        self.state.extend(
                            [item for p in listdir(f"{path}{item}")])
                    else:
                        self.imgs.extend(
                            [f"{path}{item}/{p}" for p in listdir(f"{path}{item}")])
                        self.state.extend(
                            ["other" for p in listdir(f"{path}{item}")])
        return True

    def clean(self):
        self.imgs = []
        self.other_imgs = []
        self.state = []
        self.other_state = []


    def imgs_load_other(self):
        '''
        Load the image's directions in the paths given
        '''
        if len(self.other_imgs):
            print("Images already loaded, plesase clean (clean() method) before load again.")
            return False

        for path in self.paths:
            for item in listdir(path):
                if item not in ignore:
                    if item in self.other_emotions:
                        self.other_imgs.extend(
                            [f"{path}{item}/{p}" for p in listdir(f"{path}{item}")])
                        self.other_state.extend(
                            [item for p in listdir(f"{path}{item}")])  
        return True


    def image_to_data(self,group='initial'):
        '''
        Load the data of images in a array.
        '''
        if group != 'initial':
            imgs = self.other_imgs
        else:
            imgs = self.imgs
        try:
            imgs_ = []
            for p in imgs:
                temp = Image.open(p)
                save = temp.copy()
                imgs_.append(save)
                temp.close()
        except:
            print(f'fail proccessing {p}')
            self.state.pop(self.imgs.index(p))
            #hay que buscar el indice de esa foto y quitarla de estate-
        if group != 'initial':
            self.other_imgs = imgs_
        else:
            self.imgs = imgs_



    def data_conversion(self,group='initial'):
        '''
        Convert in gray images with dimension provided.
        '''
        if group != 'initial':
            imgs = self.other_imgs
        else:
            imgs = self.imgs

        imgs_array = []
        for f in imgs:
            print(f)
            img = f.convert("L").resize(self.dim)
            imgs_array.append(np.array(img))

        imgs_array = [el/255 for el in imgs_array]
        imgs_array = np.array(imgs_array)
        imgs_array= imgs_array.reshape((len(imgs_array),48,48,1))

        if group != 'initial':
            self.other_imgs =  imgs_array
        else:
            self.imgs =  imgs_array

    