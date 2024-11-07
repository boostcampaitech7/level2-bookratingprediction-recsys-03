import argparse
import ast
from omegaconf import OmegaConf
import pandas as pd
from src.utils import Logger, Setting
import src.data as data_module
from src.train import train, test
import src.models as model_module
import optuna
from sklearn.metrics import root_mean_squared_error

def objective(trial, args, data):
    if args.model in ['CatBoost', 'XGBoost', 'LightGBM']:
        p = args.optuna_args[args.model]
        params = {
            param['name']: (
                trial.suggest_int(param['name'], param['min'], param['max']) if param.type == 'int' else 
                trial.suggest_float(param['name'], param['min'], param['max'])
            ) for _, param in p.items()
        }
        if args.model == 'CatBoost':
            params['verbose'] = False
            params['task_type'] = 'GPU'
            params['devices'] = '0'
            params['cat_features'] = [i for i in range(12)]
        else:
            params['device'] = 'cuda'

        model = getattr(model_module, args.model)(**params)

        # Prepare data for CatBoost
        train_data = data['train_dataloader'].dataset
        if args.device == 'cuda':
            X_train, y_train = train_data[:][0].numpy(), train_data[:][1].numpy()
        else:
            X_train, y_train = train_data[:][0].cpu().numpy(), train_data[:][1].cpu().numpy()
        
        valid_data = data['valid_dataloader'].dataset
        if args.device == 'cuda':
            X_valid, y_valid = valid_data[:][0].numpy(), valid_data[:][1].numpy()
        else:
            X_valid, y_valid = valid_data[:][0].cpu().numpy(), valid_data[:][1].cpu().numpy()

        model.fit(X_train, y_train)
        y_hat = model.predict(X_valid)
        y_hat_train = model.predict(X_train)
        train_rmse = root_mean_squared_error(y_train, y_hat_train)
        valid_rmse = root_mean_squared_error(y_valid, y_hat)
        trial.set_user_attr('train_rmse', train_rmse)

        return valid_rmse
    

def main(args):
    Setting.seed_everything(args.seed)

        ######################## LOAD DATA
    datatype = args.model_args[args.model].datatype
    data_load_fn = getattr(data_module, f'{datatype}_data_load')  # e.g. basic_data_load()
    data_split_fn = getattr(data_module, f'{datatype}_data_split')  # e.g. basic_data_split()
    data_loader_fn = getattr(data_module, f'{datatype}_data_loader')  # e.g. basic_data_loader()
    
    print(f'--------------- {args.model} Load Data ---------------')
    data = data_load_fn(args)
    print(f'--------------- {args.model} Train/Valid Split ---------------')
    data = data_split_fn(args, data)
    data = data_loader_fn(args, data)

    ####################### Setting for Log
    setting = Setting()
    
    if args.predict == False:
        log_path = setting.get_log_path(args)
        logger = Logger(args, log_path)
        logger.save_args()
    
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(lambda trial: objective(trial, args, data), n_trials=args.optuna_trials)

    # wandb log 생성용 모델
    print(f'-------------------- WANDB LOG FILE ---------------------')
    model = getattr(model_module, args.model)(**study.best_params)
    model = train(args, model, data, logger, setting)
    if args.wandb:
        wandb.log({'best_params': study.best_params}) 

    # 최종 제출용 모델
    print(f'----------------- {args.model} PREDICT -----------------')
    submit_model = getattr(model_module, args.model)(**study.best_params)
    submit_model.fit(data['train'].drop('rating', axis=1), data['train']['rating'])
    predicts = test(args, submit_model, data, setting, args.checkpoint)

    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')
    submission['rating'] = predicts

    filename = setting.get_submit_filename(args)
    print(f'Save Predict: {filename}')
    submission.to_csv(filename, index=False)


if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    

    arg = parser.add_argument
    str2dict = lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}

    # add basic arguments (no default value)
    arg('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', required=True)
    arg('--predict', '-p', '--p', '--pred', type=ast.literal_eval, 
        help='학습을 생략할지 여부를 설정할 수 있습니다.')
    arg('--checkpoint', '-ckpt', '--ckpt', type=str, 
        help='학습을 생략할 때 사용할 모델을 설정할 수 있습니다. 단, 하이퍼파라미터 세팅을 모두 정확하게 입력해야 합니다.')
    arg('--model', '-m', '--m', type=str, 
        choices=['FM', 'FFM', 'DeepFM', 'NCF', 'WDN', 'DCN', 'Image_FM', 'Image_DeepFM', 'Text_FM', 'Text_DeepFM', 'ResNet_DeepFM', 'XGBoost', 'LightGBM', 'CatBoost'],
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--seed', '-s', '--s', type=int,
        help='데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.')
    arg('--device', '-d', '--d', type=str, 
        choices=['cuda', 'cpu', 'mps'], help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--wandb', '--w', '-w', type=bool,
        help='wandb를 사용할지 여부를 설정할 수 있습니다.')
    arg('--wandb_project', '--wp', '-wp', type=str,
        help='wandb 프로젝트 이름을 설정할 수 있습니다.')
    arg('--run_name', '--rn', '-rn', '--r', '-r', type=str,
        help='wandb에서 사용할 run 이름을 설정할 수 있습니다.')
    arg('--model_args', '--ma', '-ma', type=ast.literal_eval)
    arg('--dataloader', '--dl', '-dl', type=ast.literal_eval)
    arg('--dataset', '--dset', '-dset', type=ast.literal_eval)
    arg('--optimizer', '-opt', '--opt', type=ast.literal_eval)
    arg('--loss', '-l', '--l', type=str)
    arg('--lr_scheduler', '-lr', '--lr', type=ast.literal_eval)
    arg('--metrics', '-met', '--met', type=ast.literal_eval)
    arg('--train', '-t', '--t', type=ast.literal_eval)

    
    args = parser.parse_args()


    ######################## Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]

    # Configuration 콘솔에 출력
    print(OmegaConf.to_yaml(config_yaml))
    
    ######################## W&B
    if args.wandb:
        import wandb
        # wandb.require("core")
        # https://docs.wandb.ai/ref/python/init 참고
        wandb.init(project=config_yaml.wandb_project, 
                   config=OmegaConf.to_container(config_yaml, resolve=True),
                   name=config_yaml.run_name if config_yaml.run_name else None,
                   #tags=[config_yaml.model],
                   entity="remember-us",
                   resume="allow")
        config_yaml.run_href = wandb.run.get_url()

        wandb.run.log_code("./src")  # src 내의 모든 파일을 업로드. Artifacts에서 확인 가능

    ######################## MAIN
    main(config_yaml)

    if args.wandb:
        wandb.finish()