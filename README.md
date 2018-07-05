# 機器學習導論

## Face Recognizer System for ML Term Project

Members: 許瑀玹 410321114 龔柏森 410321141 李婉嘉 410321156

## 1.	The ML technology you used in the recognizer
語言:

Python: 專案主要的撰寫語言

Shell Script: 將資料做成list分別存成test.txt與train.txt

套件:

Tkinter: 主要來製作GUI

PIL: GUI中的圖像處裡

Keras: 運用於專案中的圖像處裡 人臉辨識(vgg)與資料存取

vggface: 抓特徵值

vgg16: 卷積神經網路，人臉辨識

## 2.	The modules enclosed in your recognizer and their functions

模組:

Filedialog: GUI中獲取文件路徑

Numpy: 通過在指定位置插入新的軸來擴展數組形狀

h5py: 創建文件

os: 文件目錄處裡

## 3. How you test your recognizer to evaluate its recognition rate

我們使用的是VGGface來抓取特徵值，在和VGG16的人臉辨識做搭配。

使用Sequential模組進行訓練及評估識別率。

## 4.	The problems suffered in your development

一開始我們的識別率一直都偏低:

![image](https://github.com/gumball0629/Final-Project/blob/master/FP_IML_Data/Screenshot5.png)

之後將 epochs 從10改成50後，識別率就大大的提升:

也可以從數據中看到，識別率慢慢的在增加，過程中有一些浮動

![image](https://github.com/gumball0629/Final-Project/blob/master/FP_IML_Data/Screenshot4.png)

![image](https://github.com/gumball0629/Final-Project/blob/master/FP_IML_Data/Screenshot3.png)

## 5. The task allocation of each member and how you integrate the task outputs from all members

許瑀玹 410321114: Data Training, Data Testing, Version Control

龔柏森 410321141: GUI製作, Data Input, Version Control

李婉嘉 410321156: System Documentation, Version Control

## 6. Any bonus features or functionalities included in your recognizer

我們做了GUI: 

![image](https://github.com/gumball0629/Final-Project/blob/master/FP_IML_Data/Screenshot2.png)

## 7. How you feel about doing this project

我們從這個做中學到了，VGGface 與 VGG16 的運用，也學到如何將檔案打包成exe檔

## 參考資料

[VGG] (https://ithelp.ithome.com.tw/articles/10192162)









