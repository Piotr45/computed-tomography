import numpy as np
import matplotlib.pyplot as plt


def patient_summary(ds):
    print("Imię i naziwsko: ", ds.PatientName)
    print("ID pacjenta: ", ds.PatientID)
    print("Komentarz: ", ds.ImageComments)

def interactive_imgs(image, results, sinograms, x=0):
    f, ax = plt.subplots(1,3)
    f.set_figheight(15)
    f.set_figwidth(15)
    ax[0].imshow(image, cmap='bone')    
    ax[1].imshow(results[x], cmap='bone')    
    ax[2].imshow(sinograms[x], cmap='bone')
    
    for x in ax:
        x.axis('off')