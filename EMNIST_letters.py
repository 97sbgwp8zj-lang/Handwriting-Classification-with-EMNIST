#imports all of the necessary elements for the code
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import string

#loads EMNIST letters dataset and splits it into training and testing sets
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    as_supervised = True, #returns data as (image, label) pairs
    with_info= True #also returns metadata
)

#converts data from TensorFlow Tensors to NumPY arrays so the machine can learn from them
#looks over dataset and collects all images and labels into separate lists
#returns two numPy arrays: one for images, one for labels
def preprocess(ds):
    images, labels = [], []
    for image, label in tfds.as_numpy(ds):
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)


#applies preprocessing func to both datasets
x_train, y_train = preprocess(ds_train)
x_test, y_test = preprocess(ds_test)

#EMNIST dataset starts from 1 for A, so we need to subtract one to make them
#start at zero
y_train -= 1
y_test -= 1


#scales images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32')/255.0



#builds model 
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(26)
])


#loss function, uses Adam optimizer and tracks accuracy
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

#trains model with 40 epochs (complete passes through dataset)
model.fit(x_train, y_train, epochs=40)

#tests model on unseen data and prints accuracy/loss
model.evaluate(x_test, y_test, verbose=2)


probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])


#creates GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMNIST Letter Classifier")

        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack()

        self.button = tk.Button(self, text="Predict", command=self.predict_digit)
        self.button.pack()

        self.clear_button = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.result_label = tk.Label(self, text="Draw a digit and click Predict", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new("L", (280, 280), color=255)
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw_image.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_image.rectangle([0, 0, 280, 280], fill=255)
        self.result_label.config(text="Canvas cleared. Draw a digit.")

    def predict_digit(self):
        letters = list(string.ascii_uppercase)
        
      
        resized = self.image.resize((28, 28))
        inverted = ImageOps.invert(resized)
        inverted =  inverted.rotate(90)
        inverted = inverted.transpose(Image.FLIP_TOP_BOTTOM)
        img_array = np.array(inverted) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)  
        
        prediction = probability_model.predict(img_array)
        digit = np.argmax(prediction)  
        confidence = np.max(prediction)  
  
        predicted_letter = letters[digit]
        
        self.result_label.config(text=f"Predicted letter: {predicted_letter} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    app = App()
    app.mainloop()
