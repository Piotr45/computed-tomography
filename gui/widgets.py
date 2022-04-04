import numpy as np
import matplotlib.pyplot as plt


def patient_summary(ds):
    print("Imię i naziwsko: ", ds.PatientName)
    print("ID pacjenta: ", ds.PatientID)
    print("Komentarz: ", ds.ImageComments)


def show_images(original, sinogram, reconstructed, cmap='gray'):
    f, ax = plt.subplots(1, 3)
    f.set_figheight(15)
    f.set_figwidth(15)

    ax[0].imshow(original, cmap=cmap)
    ax[1].imshow(sinogram, cmap=cmap)
    ax[2].imshow(reconstructed, cmap=cmap)

    for xx in ax:
        xx.axis('off')


def interactive_imgs(image, results, sinograms, x=0):
    show_images(image, sinograms[x], results[x])

    print(rmse(image, results[x]))


def rmse(img, img_rec):
    return np.sqrt(np.mean(np.square(img - img_rec)))