# 1, data prepare
you need have  scRNA-sequencecing data,GRN-network dataset and TF¡¢target dataset.We provide public datasets in the data folder
># 2, Code environment
python==3.9
dgl=1.1.2
torch==2.1.0
numpy==1.26.3
scipy==1.11.4
networkx==3.1
sklearn==1.3.0
># 3,Usage
3.1 Partition the data set
    `code/train_spilt.py`# use this code to divide the training set, test set, and validation set
3.2 Traing
    "code/GeneRegLink.py"#Model training and testing performance

 
