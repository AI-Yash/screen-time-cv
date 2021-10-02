
# AI Screen Time Measurement System

In this pandemic our screen time has increased, measuring it becomes crucial, we have apps for that, but they can't measure the time of every individual using the device, to solve this we have created a screen time measurement system which is powered by AI. This system can accurately measure screen time of multiple users, and output that time in a database.

### [website](https://sites.google.com/view/ai-screen-time/home)


[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/yashppawar/screen-time-cv/blob/fr2/LICENSE)
 
## Authors

[@yashppawar](https://www.github.com/yashppawar) and [@YashVardhan-AI](https://www.github.com/porus-creator)

  
## Installation

Install AI Screen Time Management system

requirements:
- python (>3.7)
- git (only for installation)

clone the repo
```bash
  git clone git@github.com:yashppawar/screen-time-cv.git
```
or 
```bash
  git clone https://github.com/yashppawar/screen-time-cv.git
```
**Install the required libraries**
```bash
  cd screen-time-cv
  pip install -r requirements.txt
```
or 
```bash
  cd screen-time-cv
  pip3 install -r requirements.txt
```
on linux and mac
## Documentation

### How to add members?
 To add members you will have to add images of that person's face in the dataset folder.
 
 The images should be inside a folder named after the name of the person, it should be like the below structure

 ```fs
 .
 ├──dataset
 │ ├──Person One
 │ │ ├──1.jpg
 │ │ ├──2.jpg
 │ │ ├──3.jpg
 │ │ ├──4.jpg
 │ │ ├──5.jpg
 │ │ └──6.jpg
 │ │ ...
 │ ├──Person Two
 │ │ ├──1.jpg
 │ │ ├──2.jpg
 │ │ ├──3.jpg
 │ │ ├──4.jpg
 │ │ ├──5.jpg
 │ │ └──6.jpg
 │ │  ...
 │ ├──...    
```

 ### running the app

```
 python ./"test from video.py"
```
or
```
 python3 ./"test from video.py"
```
