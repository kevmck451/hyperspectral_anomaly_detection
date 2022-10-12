# App for inspecting images for analysis

import PySimpleGUI as sg

icon = 'Images/Recon Icon.png'


layout = [[sg.Text("Hello from PySimpleGUI")], [sg.Button("OK")]]

# Create the window
window = sg.Window('RECON', layout, margins=(800,500))

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "OK" or event == sg.WIN_CLOSED:
        break

window.close()