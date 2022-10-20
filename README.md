# NLP-Begin
In this repo, I'm learning and doing NLP from scratch

## Update work 2022-10-20
**Step Training**

	- Using PhoBERT pretrain model
	- Runing with batchsize = 4, epoch = 15
	- Model is converge at epoch 12, with F1 = 1.00 and Acc = 1.00 :)
	- Data of each class is not balance, so that the above result quite ambiguous

## Update work 2022-10-13
**Step Preprocessing**

	- Using Pyvi instead of Underthesea in Word segmentation step
	- Completing pipeline preprocessing
	- Fix error the character "/" was removed after using normalized accents
	- Cleaning data before labeling

## Update work 2022-10-12
**Step Preprocessing**

	- Removing hashtag
	- Removing URL/HTML code
	- Removing emoji
	- Removing extra spaces
	- Word segmentation using Underthesea
