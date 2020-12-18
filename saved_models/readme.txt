When you run train.py and specify a folder to save your model to (like: python3 train.py --run_folder=mytestrun --epoch_start=0 epoch_end=15 --batch_size=64 --learning_rate=0.01) a folder is created under saved_models
(in this example it will be mytestrun).
For each epoch a subfolder with epoch number is created.
Thus in our example after epoch #4 is over, the current training state of the model will be saved to saved_models/mytestrun/4/.
If you don't see anything in this folder yet, that's OK.
It will start populating after you run training.  