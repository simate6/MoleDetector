import PIL
from PIL import Image, ImageTk
import cv2
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.models import load_model

# load model
model = load_model('cnn_weights.hdf5')

# capture
width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def show_frame():
    global frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


def takephoto():
    cv2.imwrite('Test/test.png', frame)
    find()


def close():
    cap.release()
    root.quit()
    root.destroy()


def find():
    shift = np.array([10,20,40], dtype=np.uint8) #window shifted by number of pixels
    size = np.array([32,64,128], dtype=np.uint8) #window sizes in pixels

    uniform_size = 32 #input size of the classificator CNN
    i = 0

    for i_shift, i_size in zip(shift, size): #linking the two jointly changed parameters
        for step_y in range(0,3): #shifting the slicing grid in y direction
            for step_x in range(0,3): #shifting the slicing grid in x direction
                diagimg = np.asarray(PIL.Image.open('Test/test.png')) #reading the currently diagnosed input image
                diagimg = diagimg[i_shift*step_y:, i_shift*step_x:]  #shifting the image according to the corresponding parameters
                grid = diagimg[0:(diagimg.shape[0]//i_size)*i_size, 0:(diagimg.shape[1]//i_size)*i_size] #cropping the image to fit even windows
                grid_sliced = np.asarray(np.split(grid, grid.shape[0]//i_size, axis = 0)) #slicing the image in y direction
                grid_sliced = np.asarray(np.split(grid_sliced, grid_sliced.shape[2]//i_size, axis = 2)) #slicing the image in x direction
                grid_sliced = np.reshape(grid_sliced, (grid_sliced.shape[0]*grid_sliced.shape[1], i_size, i_size, 3)) #reshaping the image into a column array each row containing a sliced window

                vector_x = np.arange(i_shift*step_x, grid.shape[1], i_size) #getting the window positions in x direction
                vector_y = np.arange(i_shift*step_y, grid.shape[0], i_size) #getting the window positions in y direction
                matrix = np.asarray(np.meshgrid(vector_x, vector_y, sparse=False, indexing='ij')) #creating the window position matrix
                info_img = np.transpose(np.reshape(matrix, (2,matrix.shape[1]*matrix.shape[2]))) #reshaping the window position matrix into a column array, each row containing the position of the correspondig sliced window
                info_img = np.concatenate((info_img, np.multiply(i_size,np.ones((info_img.shape[0],1)))), axis=1) #adding the size of the windows to the column array as plus information

                if i_size > uniform_size:
                    out_shape = grid_sliced.shape[0], uniform_size, grid_sliced.shape[1]//uniform_size, uniform_size, grid_sliced.shape[2]//uniform_size, 3 #defining the expected image size for the CNN
                    grid_sliced = grid_sliced.reshape(out_shape).mean(2).mean(3) #resizing the images according to the expected shape for the CNN
                if i==0:
                    input_sequence = grid_sliced
                    info = info_img
                    i += 1
                else:
                    input_sequence = np.append(input_sequence, grid_sliced, axis = 0) #extending the input_sequence with the current new inputs
                    info = np.vstack((info,info_img)) #extending the information array with the current new information
                    i += 1

    info[:, 0], info[:, 1] = info[:, 1], info[:, 0].copy() #changing the order of the x and y positions of the windows

    input_sequence_scaled = input_sequence/255 #normalizing the input images

    # predict & draw
    preds = model.predict(input_sequence_scaled)
    diagimg = np.asarray(PIL.Image.open('Test/test.png'))

    # figure = plt.figure()
    fig, ax = plt.subplots(1, figsize=(10, 10), dpi=100)
    ax.imshow(diagimg)

    for i in range(0,preds.shape[0]):
        if preds[i, 1] > 0.5:
            rect = patches.Rectangle((info[i,1],info[i,0]),info[i,2],info[i,2],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    root = Tk()
    root.bind('<Escape>', lambda e: close())
    lmain = Label(root)
    lmain.grid(sticky="WENS")
    tp_button = Button(root, text="Take photo", bg="blue", fg="white",
                       command=lambda: takephoto())
    tp_button.grid(row=1, sticky="WE")

    show_frame()
    root.mainloop()
