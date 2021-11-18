# HoN
FER Machine learning project to detect people's emotions at a certain moment in different environments

# Requirements
This libraries are needed:
### Making models and predictions:
    - Tensorflow
    - OpenCV
    - Numpy
    - PIL
    - os
### UI 
    - PyQt6
    - sys
    - mms
    - CamGear
    - OpenCV

### UI design
    - QtDesigner

# Installation   
Once we have these libraries, we can compile the graphical environment with:

        pyuic6 -o hon.py main.ui

and start the program with:

        python main_gray.py

# Contents   
### clean&fit
    In the clean & fit directory you will find all the algorithms and functions used to load, review, clean and transform the datasets.

##### Landmarks model
    The first attempt was to fit a model and make predictions only with the landmarks of each photo.

<img src="imgs/lmhap.png"
     width= "128px" /> 
<img src="imgs/lmamger.png"
     width= "128px" /> 
<img src="imgs/lmdis.png"
     width= "128px" />
<img src="imgs/lmfear.png"
     width= "128px" />    
    
    This model had a very low accuracy after testing with several datasets, therefore it was discarded.

##### Images models
    I did model training with various datasets (see the bottom of the page) performing different treatments of the images as well as combinations of models.

    Finally I used to load in the UI a model with 7 recognizable emotions and an accuraxy of about 60%.


### main
    Here we can find the files of the interface created with qtdesigner.

    We can analyze images from videos from three different sources:
    - Webcam
    - Youtube
    - Screenshot

#### Webcam
[![WebCam Capture](https://img.youtube.com/vi/jKjpwnIFuek/0.jpg)](https://youtu.be/jKjpwnIFuek)

#### Youtube

[![Youtube Capture](https://img.youtube.com/vi/WwIlMd8oUkU/0.jpg)](https://youtu.be/WwIlMd8oUkU)
#### Screenshot

[![Screen Capture](https://img.youtube.com/vi/CkPy4aHnup8/0.jpg)](https://youtu.be/CkPy4aHnup8)









## Used data & resources

[1] Kosti, Ronak, Jose M. Alvarez, Adria Recasens, and Agata Lapedriza. "Context based emotion recognition using emotic dataset." IEEE transactions on pattern analysis and machine intelligence 42, no. 11 (2019): 2755-2766.
[2] Kosti, Ronak, Jose M. Alvarez, Adria Recasens, and Agata Lapedriza. "Emotion recognition in context." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1667-1675. 2017.
[3] Kosti, Ronak, Jose M. Alvarez, Adria Recasens, and Agata Lapedriza. "EMOTIC: Emotions in Context dataset." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 61-69. 2017.

----------------------------------------

FER Dataset
https://www.kaggle.com/msambare/fer2013

---------------------------------------

P. Lucey, J. F. Cohn, T. Kanade, J. Saragih, Z. Ambadar and I. Matthews, "The Extended Cohn-Kanade Dataset (CK+): A complete dataset for action unit and emotion-specified expression," 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition - Workshops, 2010, pp. 94-101, doi: 10.1109/CVPRW.2010.5543262.

---------------------------------

TFEID Dataset

-----------------------------

FACES https://faces.mpdl.mpg.de/imeji/

----------------------------

SENSA EMOJIS https://sensa.co/emoji/