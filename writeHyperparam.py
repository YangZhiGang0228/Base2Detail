import json
import os

def main():

    data = {
        "train_pkl": "./data_PKL/cartoon_elephant_train.pkl",
        "valid_pkl": "./data_PKL/cartoon_elephant_valid.pkl",
        "output_path": './jobs/net_cartoon_elephant/',
        "epochs": 10000, 
        "lr": 2e-3, 
        "device": 'cuda',
        "Din": 6,
        "Dout": 32,
        "Lifted": 256,
        "h_initNet": [32,32],
        "h_liftNet": [64,128],
        "h_matrixNet": [32,32],
        "h_FNet": [128,64],
        "numSubd": 3,
    }

    # create directory
    if not os.path.exists(data['output_path']):
        os.mkdir(data['output_path'])

    # write hyper parameters into a json file
    with open(data['output_path'] + 'hyperparameters.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()