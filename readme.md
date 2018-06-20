## ECG classification

#### Requirements
* tensorflow
* numpy
* scipy
* pandas  
Also, you can use the command `pip3 install -r requirements.txt` to install the dependency packages.  
In this project, both python2 and python3 are ok(But we strongly suggest that you use python3).

#### How to Run
1. Put the data set in folder.
2. Run `merge_dataset.py` to create **train.mat** and **test.mat**. Use the following command to run the code.    
```
python3 merge_dataset.py --dir YOUR_TRAINING_SET_FOLDER_NAME
```  
Use `python3 merge_dataset.py -h` if you need some help.    
3. Run `train.py`. You can choose your parameter for the following parameters in your command.  
   * learning_rate 
   * epochs
   * batch_size.
   * k_folder: True/False.   

   If you want to begin the process for k-folder validation, use the following command: `python3 train.py --k_folder True`. If you only want to train the model, use the command: `python3 train.py`.
Use `python3 train.py -h` if you need some help.  
   
4. After you train the model, use `test.py` to test the accuracy and F1 rate. The default path for checkpoints is **checkpoints/**. If you use other path, run the test.py use the following command:
```
python3 test.py --check_point_folder YOUR_CHECKPOINT_FOLDER_PATH
```


#### Experiment result
The F1 for our model is **0.82**. But maybe if you run you will get a different number for that the training and testing set is randomly choose.

