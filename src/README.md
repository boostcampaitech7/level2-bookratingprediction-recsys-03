## 🌳 File Tree 🌳


```
📂 src
├── 📂 data
│   ├── __init__.py 
│   ├── basic_data.py 
│   ├── context_data.py 
│   ├── image_data.py 
│   └── text_data.py
│
├── 📂 ensemble
│   └── ensembles.py
│
├── 📂 loss
│   ├── __init__.py
│   └── loss.py
│
├── 📂 models
│   ├── .DS_Store
│   ├── __init__.py 
│   ├── _helper.py
│   ├── image_FM.py
│   └── text_FM.py
│
├── 📂 train
│   ├── __init__.py 
│   └── trainer.py
│
├── __init__.py
├── utils.py
└── README.md
```


## Structure

### data

- **`basic_data.py`**
    
    기본적인 정형 데이터를 전처리를 수행합니다.
    
    - `basic_data_load(args)` : 학습/테스트 데이터가 포함된 딕셔너리를 반환합니다.
    - `basic_data_split(args, data)` : `basic_data_load` 에서 생성된 딕셔너리를 입력으로 받아 학습 데이터를 train/valid 데이터셋으로 나누어 추가한 뒤 반환합니다.
    - `basic_data_loader(args, data)` : `basic_data_split` 으로 반환된 딕셔너리의 데이터셋을 DataLoader로 변환해 추가한 뒤 반환합니다.
- **`context_data.py`**
    
    문맥 데이터를 포함하여 전처리를 수행합니다.
    
    - `str2list(x: str)` : str type으로 이루어진 리스트를 list type으로 변환하는 함수입니다.
    - `split_location(x: str)` : location 데이터를 국가, 주, 시 등으로 나누어 정제합니다.
    - `text_preprocessing(summary: str)` : 주어진 텍스트를 다음 규칙에 따라 전처리합니다.
        1. 특수 문자 제거
        2. 알파벳과 숫자, 공백을 제외한 모든 문자 제거
        3. 여러 개의 공백을 하나의 공백으로
        4. 문자열의 앞뒤 공백 제거
        5. 모든 문자를 소문자로 변환
    - `categorize_publication(x: int, a: int)` : 주어진 연도를 아래의 기준에 따라 카테고리화합니다.
        1. 1970년 이하의 연도는 1970으로 반환
        2. 2000년 초과의 연도는 2006으로 반환
        3. 나머지 연도는 a 값에 맞게 그룹화하여 반환
    - `extract_language_from_isbn(isbn: str)` : ISBN 정보를 입력받아 언어 코드를 추출합니다.
    - `replace_language_using_isbn(books: pd.DataFrame)` : language 데이터의 결측치에 대해 ISBN 정보를 활용해 채웁니다.
    - `categorize_age(x: int, a: int)` : 주어진 나이 정보를 다음 규칙에 따라 카테고리화합니다.
        1. 20세 미만은 10으로 변환
        2. 60세 이상은 60으로 변환
        3. 나머지는 나이대에 맞게 변환
    - `process_context_data(users: pd.DataFrame, books: pd.DataFrame)` : 위의 함수들을 적용하여 문맥 데이터를 전처리합니다.
    - `context_data_load(args)` : 학습/테스트 데이터가 포함된 딕셔너리를 반환합니다.
    - `context_data_split(args, data)` : `basic_data_load` 에서 생성된 딕셔너리를 입력으로 받아 학습 데이터를 train/valid 데이터셋으로 나누어 추가한 뒤 반환합니다.
    - `context_data_loader(args, data)` : `basic_data_split` 으로 반환된 딕셔너리의 데이터셋을 DataLoader로 변환해 추가한 뒤 반환합니다.
- **`image_data.py`**
    
    이미지 데이터를 벡터화하여 추가하는 파일입니다.
    
    - `Image_Dataset(Dataset)` : 벡터화된 이미지 데이터를 저장하는 클래스입니다.
    - `image_vector(path, imag_size)` : 이미지 경로를 입력으로 받아 벡터화하여 반환합니다.
    - `process_img_data(users, books, args)` : 이미지 데이터를 벡터화하여 추가한 데이터프레임을 반환합니다.
    - `image_data_load(args)` : 학습/테스트 데이터가 포함된 딕셔너리를 반환합니다.
    - `image_data_split(args, data)` : `image_data_load` 에서 생성된 딕셔너리를 입력으로 받아 학습 데이터를 train/valid 데이터셋으로 나누어 추가한 뒤 반환합니다.
    - `image_data_loader(args, data)` : `image_data_split` 으로 반환된 딕셔너리의 데이터셋을 DataLoader로 변환해 추가한 뒤 반환합니다.
- **`text_data.py`**
    
    텍스트 데이터를 벡터화하여 추가하는 파일입니다.
    
    - `text_preprocessing(summary)` : 텍스트 데이터를 입력받아 정규화 등의 기본적인 전처리를 수행하여 반환합니다.
    - `text_to_vector(text, tokenizer, model)` : 텍스트 데이터를 사전학습된 언어 모델을 이용해 임베딩하여 반환합니다.
    - `process_text_data(ratings, users, books, tokenizer, model, vector_create=False)` : 유저/책 정보가 포함된 데이터프레임을 입력받아 유저가 읽은 책에 대한 요약 정보와 책의 고유 요약 정보를 벡터화해 변수로 추가한 데이터프레임을 반환합니다.
    - `Text_Dataset(Dataset)` : 벡터화된 텍스트 데이터를 저장하는 데이터셋 클래스입니다.
    - `text_data_load(args)` : 학습/테스트 데이터가 포함된 딕셔너리를 반환합니다.
    - `text_data_split(args, data)` : `text_data_load` 에서 생성된 딕셔너리를 입력으로 받아 학습 데이터를 train/valid 데이터셋으로 나누어 추가한 뒤 반환합니다.
    - `text_data_loader(args, data)` : `text_data_split` 으로 반환된 딕셔너리의 데이터셋을 DataLoader로 변환해 추가한 뒤 반환합니다.

### ensemble

- **`ensembles.py`**
    
    다양한 방식으로 앙상블(가중치, 혼합 등)하는 기능을 제공하는 모듈입니다.
    
    - `Ensemble` : 앙상블을 진행하는 클래스입니다.
    - `simple_weighted(self, weight:list)` : 직접 weight를 지정하여 앙상블합니다.
    - `average_weighted(self)` : 모든 모델에 동일한 weight(1/n)로 앙상블을 진행합니다.
    - `mixed(self)` : Negative case 발생 시 다음 순서에서 예측한 rating으로 넘어가서 앙상블합니다.

### loss

- **`loss.py`**
    
    RMSE 손실을 계산하기 위한 모듈입니다.
    
    - `RMSELoss(nn.module)` : Pytorch의 nn.module을 상속받아 구현된 클래스입니다. RMSE 손실을 계산하는데 사용됩니다.
        - `forward(x, y)` : 예측값 x와 실제값 y를 입력으로 받아 MSE 손실을 계산하고, 제곱근을 적용하여 RMSE 손실을 반환합니다.

### models

- **`image_FM.py`**
    
    기존 유저/상품 벡터와 이미지 벡터를 결합하여 FM 기반 모델을 모아놓은 모듈입니다.
    
    - `Image_FM(nn.Module)` : Pytorch의 nn.module을 상속받아 구현된 클래스입니다. 기존 유저/상품 벡터와 이미지 벡터를 결합하여 FM으로 학습하는 모델을 구현합니다.
        - `forward(self, x)`: 사용자-책 및 이미지 데이터를 입력받아 첫 번째 및 두 번째 상호작용을 계산하고 최종 출력을 반환하는 순전파 메서드입니다.
    - `Image_DeepFM(nn.Module)` : Pytorch의 nn.module을 상속받아 구현된 클래스입니다. 사용자-책 및 이미지 데이터를 입력받아 첫 번째 및 두 번째 상호작용을 계산하고, Deep FM을 통해 추가적인 특징 학습을 수행하여 최종 출력을 반환하는 모델을 구현합니다.
        - `forward(self, x)` : 사용자-책 및 이미지 데이터를 입력받아 첫 번째 및 두 번째 상호작용을 계산하고, Deep Neural Network을 통해 추가적인 특징을 학습하여 최종 출력을 반환하는 순전파 메서드입니다.
    - `ResNet_DeepFM(nn.Module)` : Pytorch의 nn.module을 상속받아 구현된 클래스입니다. FM과 Deep Neural Network를 결합하여 유저/상품 벡터와 ResNet을 통해 임베딩된 이미지 벡터를 학습하는 모델을 구현합니다.
        - `forward(self, x)` : FM과 Deep Neural Network를 통해 학습한 결과를 결합해 최종 출력을 반환하는 순전파 메서드입니다.
- **`text_FM.py`**
    
    유저와 상품의 sparse 및 dense feature를 결합하여 FM 기반 모델을 모아놓은 모듈입니다.
    
    - `Text_FM(nn.Module)` : Pytorch의 nn.module을 상속받아 구현된 클래스입니다. 기존 유저/상품 벡터와 텍스트 벡터를 결합해 FM으로 학습하는 모델을 구현합니다.
        - `forward(self, x)` : 입력된 유저/상품 및 텍스트 벡터를 사용해 first-order 및 second-order 상호작용을 계산하고 최종 출력을 반환하는 순전파 메서드입니다.
    - `Text_DeepFM(nn.Module)` : Pytorch의 nn.module을 상속받아 구현된 클래스입니다. FM과 Deep Neural Network를 결합하여 sparse 및 dense feature를 함께 학습하는 모델을 구현합니다.
        - `forward(self, x)` : 입력된 유저/상품 및 텍스트 벡터로부터 첫 번째 및 두 번째 상호작용을 계산하고, Deep Neural Network을 통해 추가적인 특징을 학습하여 최종 출력을 반환하는 순전파 메서드입니다.

### train

- **`trainer.py`**
    
    모델 학습 및 추론한 예측값을 반환하는 기능들을 모아놓은 모듈입니다.
    
    - `train(args, model, dataloader: Dict[str, DataLoader], logger, setting)`: 훈련 데이터셋을 활용하여 모델을 학습하고 학습한 모델을 저장 및 반환합니다. (WandB 사용하는 경우 WandB에 손실값을 기록)
    - `valid(args, model, dataloader, loss_fn)`: 검증 데이터셋을 활용하여 모델의 예측을 수행하고 손실 값을 반환합니다.
    - `test(args, model, dataloader, setting, checkpoint=None)`: 테스트 데이터셋에 대한 모델의 예측값을 반환합니다. (체크포인트를 사용하는 경우 저장한 모델을 불러와 사용할 수 있습니다.)
    - `stf_train(args, model, dataloader: Dict[str, DataLoader], setting)`: Stratified K-Fold 교차 검증을 통해 모델을 학습하고 성능을 기록합니다. 테스트 데이터셋에 대한 모델의 예측값을 반환합니다.

### `utils.py`

초기 세팅, 로그 등 유틸리티 기능을 수행하는 함수를 모아놓은 모듈입니다.

- `rmse(real, predict)` : RMSE를 계산하는 함수입니다.
- `Setting` : 초기 세팅(시드, 경로)을 적용하는 클래스입니다.

    - `seed_everything(seed)` : seed 값을 고정시키는 정적메서드입니다.
    - `get_long_path(self, args)` : log file을 저장할 경로를 반환하는 메서드입니다.
    - `get_submit_filename(self, args)` : submit file을 저장할 경로를 반환하는 메서드입니다.
    - `make_dir(self, path)` : 경로가 존재하지 않을 경우 해당 경로를 생성하며, 존재할 경우 pass를 하는 메서드입니다.

- `Logger` : 로그 파일을 관리하는 클래스입니다.
    - `log(self, epoch, train_loss, valid_loss=None, valid_metrics=None)` : log file에 epoch, train loss, valid loss를 기록하는 메서드입니다. 이 때, log file은 train.log로 저장됩니다.
    - `close(self)` : log file을 닫는 메서드입니다.
    - `save_args(self)` : model에 사용된 args를 저장하는 메서드입니다. 이 때, 저장되는 파일명은 model.json으로 저장됩니다.
