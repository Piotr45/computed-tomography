from ipywidgets import IntText, interact, Checkbox, Text, VBox

style = {'description_width': 'initial'}

# load image
steps = IntText(
    value=1,
    min=1,
    max=180,
    step=1,
    description='Krok: ',
    disabled=False,
    style=style
)
number_of_detectors = IntText(
    value=90,
    min=1,
    step=1,
    description='Liczba detektorów: ',
    disabled=False,
    style=style
)
detector_distance = IntText(
    value=90,
    min=1,
    step=1,
    description='Rozpiętość między detektorami (px): ',
    disabled=False,
    style=style
)
interactive = Checkbox(
    value=True,
    description='Wersja interaktywna',
    disabled=False,
    indent=False
)
convolute = Checkbox(
    value=True,
    description='Zastosuj splot',
    disabled=False,
    indent=False
)
filename = Text(
    description='Nazwa pliku z katalogu dicom',
    style=style,
    value='shepp_logan.dcm'
)


# save image
save_filename = Text(
    description='Nazwa pliku dicom do zapisu',
    style=style,
    value='new_image.dcm'
)
patient_name = Text(
    description='Imię i nazwisko pacjenta',
    style=style,
    value='Adam Kowalski'
)
patient_id = Text(
    description='Identyfikator pacjenta',
    style=style,
    value='123456'
)
image_comments = Text(
    description='Komentarz',
    style=style,
    value='Wedlug analizy tomografii komputerowej pacjent doznal ciezkiego stluczenia czaszki'
)