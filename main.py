from email.policy import default
import argparse
import torch 
import yaml
from util.helper import seed, model_mapping

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        type=str,
                        default='train')
    parser.add_argument('--seed',
                        default=52,
                        type=int,
                        help='Seed')
    parser.add_argument('--model',
                        default='spatio_attention_embedded_rnn',
                        type=str)
    parser.add_argument('--batch_size',
                        type=int)
    parser.add_argument('--lr',
                        type=float)
    parser.add_argument('--lr_decay_ratio',
                        type=float)                        
    parser.add_argument('--hidden_dim',
                        type=int)
    parser.add_argument('--hidden_dim_2',
                        type=int)
    parser.add_argument('--dropout',
                        type=float)
    parser.add_argument('--input_len',
                        type=int)
    return parser

if __name__=="__main__":
    parser = parse_args()
    args = parser.parse_args()

    seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = model_mapping(args.model)
    with open(conf['config']) as f:
        config = yaml.safe_load(f)

    target_station = config['data']['target_station']
    config_wandb = {
        'epochs': config["train"]["epochs"],
        'patience': config['train']['patience'],
        'optimizer': config['train']['optimizer'],
        'input_len': config['model']['input_len'],
        'output_len': config['model']['output_len'],
        'train_size': config['data']['train_size'],
        'valid_size': config['data']['valid_size'],
        'batch_size': config['data']['batch_size'],
        'data_dir': config['data']['data_dir'],
        'input_features': config['data']['input_features'],
        'target_features': config['data']['target_features'],
        'num_layers': config['model']['num_layers'],
        'input_dim': config['model']['input_dim'],
        'output_dim': config['model']['output_dim'],
        'hidden_dim': config['model']['hidden_dim'],
        'hidden_dim_2': config['model']['hidden_dim_2'],
        'lr': config['train']['lr'],
        'lr_decay_ratio': config['train']['lr_decay_ratio'],
        'activation': config['model']['activation'],
        'rnn_type':  config['model']['rnn_type'],
        'dropout': config['train']['dropout'],
        'alpha': config['train']['alpha'],
        # data
        'nan_station': config['data']['nan_station'],
        'input_features': config['data']['input_features']
    }
    # test voi nhieu tram 
    for station in target_station:
        print(station)
        model = conf['model'](args, config_wandb, station, device)
        val_loss= model.train()
        model.test()

    # # test voi nhieu tram
    # model = conf['model'](args, config, )
    # if args.mode == 'train':
    #     model.train()
    # elif args.mode == 'test':
    #     model.test()
    # else:
    #     raise RuntimeError("Mode needs to be train/evaluate/test!")