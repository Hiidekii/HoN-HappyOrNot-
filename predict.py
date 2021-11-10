from tensorflow.keras.models import load_model
HEIGHT, WIDTH = 48,48
cats = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

new_model = load_model("complet_faces_best_model20211110.h5")
new_model.summary()




def get_prediction(img):
    img_arr = img.convert("L").resize((HEIGHT, WIDTH))
    img_arr = img_arr/255
    prediction = new_model.predict(img_arr)
    return cats[prediction.argmax()]
