import tkinter as tk
import os, pprint

from PIL import ImageTk, Image

import cv2 as cv, numpy as np
from keras import Sequential, models


log = lambda _type, *verbatim: print(f"[{_type}]", verbatim)

class CButton(tk.Button):
    def __init__(self, master: tk.Tk, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            font=("helvetica", 16, "bold"),
            bg="blue", bd=1
        )

class ASLLabel(tk.Label):
    def __init__(self, master: tk.Tk, **kwargs):
        super().__init__(master, **kwargs)

        self.configure(
            font=("helvetica", 32, "bold"),
            fg="black"
        )

# Interface class contains buttons under camera
class Interface(tk.Frame):
    def __init__(self, master: tk.Tk, cap: cv.VideoCapture):
        super().__init__(master=master)
        # camera will be used for button functionality
        self.cap = cap
        self.predictions = []
        self.model = models.load_model("200epochtesting.keras")
        self.alphabet = "ABCDEFGHIJKLMNO"


        self.str_pred_label = tk.StringVar(self, value=None)

        self.p_l = ASLLabel(self, textvariable=self.str_pred_label)
        self.p_l.grid(column=0, row=2 ,columnspan=3)

        # interface buttons
        # starts prediction
        self.predict_button = CButton(self, text="Predict", command=self.predict); self.predict_button.grid(column=0, row=1, pady=5)
        # quickly create training photos if necessary
        self.train_button = CButton(self, text="Train"); self.train_button.grid(column=1, row=1, pady=5)
        # take a picture for some reason
        self.capture_button = CButton(self, text="Capture", command=self.capture); self.capture_button.grid(column=2, row=1, pady=5)
        if len(self.predictions) > 0:
            self.p_l = ASLLabel(self, textvariable=self.str_pred_label, fg=None)
        
    def capture(self) -> None:
        t, f = self.cap.read()
        if t:
            len_fileregx = len([file for file in os.listdir("./") if "image_" in file])
            cv.imwrite(f"image_{len_fileregx}", f)
        else:
            log("capture", "camera could not be captured")

    
    def predict(self) -> None:
        # assert self.t == Fals, "Camera feed not active"
        t, f = self.cap.read()


        log("predict", f"pre-processed image: {f}")
        f = self._preprocess_image(frame=f)


        predictions = self.model.predict(f)        # ex: throw up A
                                                    # A: 100%
                                                    # B: 30%
                                                    # C: 10%
                                                    # ....
                                                    # E: 0.001%

        pred_class = np.argmax(predictions, axis=1)[0]
        # frame
        pred_certainty = round(np.max(predictions) * 100, 2)
        log("predict", "pred_certainty: ")
        self.predictions.append(pred_certainty * 100)
        log("predict", f"self.alphabet: {chr(pred_class+65)}")
        log("predict", f"pred_class: {pred_class}")

        # certainty_label = ASLLabel(self, textvariable=self.str_pred_label, fg=None)
        # certainty_label.grid(column=0, row=2, columnspan=3)

        if len(self.predictions) > 2:
            if self.predictions[-1] > self.predictions[-2]:
                self.p_l.configure(fg="green")
            else:
                self.p_l.configure(fg="red")
        
        if pred_certainty >= 75:
            self.str_pred_label.set(f"Letter: {chr(pred_class+65)}, Perc: {pred_certainty}%")
        else:
            self.str_pred_label.set(f"None")

    # changes self.frame (f)
    def _preprocess_image(self, frame: cv.typing.MatLike ,resize_resolution=(150, 150)) -> cv.typing.MatLike:
        if frame.ndim > 2:
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        frame = cv.resize(frame, resize_resolution)
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=-1)
        frame = np.expand_dims(frame, axis=0)

        print("processed photo: ", frame.shape)

        return frame

class UI(tk.Tk):
    def __init__(self, camera: cv.VideoCapture):
        super().__init__()

        self.title("Hello world")
        self.configure(bg="blue")
        self.camera = camera

        self.label = ASLLabel(self); self.label.grid(column=0, row=0, columnspan=3)

        self.camera_feed()
        self.interface = Interface(self, self.camera)
        self.interface.grid(column=0, row=1, columnspan=3)
        

    def camera_feed(self) -> None:
        t, f = self.camera.read()
        if t:
            frame = cv.cvtColor(f, cv.COLOR_BGR2RGB)
            frame = cv.resize(frame, (600, 300))
            image_pil: Image.Image = Image.fromarray(frame)
            image_tk = ImageTk.PhotoImage(image=image_pil)
            self.label.image_tk = image_tk
            self.label.configure(image=image_tk)
        self.label.after(10, self.camera_feed)
        

cap = cv.VideoCapture(0)

ui = UI(cap)

ui.mainloop()

cap.release()
cv.destroyAllWindows()
