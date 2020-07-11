# Introduction 
This app consists of a number of detection functionalities to infer information from videos.

# Functionalities
This app demonstrates following functionalitites:
1.	High Angle Person Detection
2.	Face Detection
3.	Facial Landmarks Detection
4.	Age & Gender Detection
5.	Emotion Detection
6.	Face Re-identification

# Installation
Run install.exe  
If the above dowsn't work, copy contents of install.ps1 and enter them in powershell

# Run
## Activate the environment: 
In cmd, run the following:  
<openvino_directory>\bin\setupvars.bat  
For example- C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat

## Run application
python interactive_face_recognition.py -i "<input_path>"  
-i arguement is optional, webcam is used by default
