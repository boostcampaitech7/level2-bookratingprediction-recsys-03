<h1 align="center"><a href='https://www.notion.so/Book-Rating-Prediction-e2b8e0c8872647b2bfdeb3f5df0cbd8a'>RecSys-03 ㄱ해줘</a></h1>
<br></br>

## 🏆 대회 개요 🏆

소비자들의 책 구매 결정에 대한 도움을 주기 위한 개인화된 상품 추천 대회, 책과 관련된 정보와 소비자의 정보, 그리고 소비자가 실제로 부여한 평점, 총 3가지의 데이터 셋을 활용하여 각 사용자가 주어진 책에 대해 얼마나 평점을 부여할지에 대해 예측한다.

- Objective : 
  **사용자가 그동안 읽은 책에 부여한 평점 데이터를 사용해서 새로운 책을 추천했을 때 어느 정도의 평점을 부여할지 예측**
- 평가 지표 : **RMSE (Root Mean Squared Error)**

<br></br>
## 👨‍👩‍👧‍👦 팀 소개 👨‍👩‍👧‍👦
    
|강성택|김다빈|김윤경|김희수|노근서|박영균|
|:--:|:--:|:--:|:--:|:--:|:--:|
|<a href='https://github.com/TaroSin'><img src='https://github.com/user-attachments/assets/75682bd3-bcff-433e-8fe5-6515a72361d6' width='200px'/></a>|<a href='https://github.com/BinnieKim'><img src='https://github.com/user-attachments/assets/ff639e97-91c9-47e1-a0c8-a5fc09c025a6' width='200px'/></a>|<a href='https://github.com/luck-kyv'><img src='https://github.com/user-attachments/assets/015ec963-d1b4-4365-91c2-d513e94c2b8a' width='200px'/></a>|<a href='https://github.com/0k8h2s5'><img src='https://github.com/user-attachments/assets/526dc87c-0122-4829-8e94-bce6f15fc068' width='200px'/></a>|<a href='https://github.com/geunsseo'><img src='https://github.com/user-attachments/assets/0a1a27c1-4c91-4fdf-b350-1540c835ee72' width='200px'/></a>|<a href='https://github.com/0-virus'><img src='https://github.com/user-attachments/assets/98470105-260e-443d-8592-c139d7918b5e' width='200px'/></a>|

<br></br>

## 🌳 File Tree 🌳

```
{level2-competitiveds-recsys-03}
|
├──📁 EDA
|   ├── davin_EDA.ipynb
|   ├── gs_EDA.ipynb
|   ├── hs_EDA.ipynb
|   ├── tarosin_EDA.ipynb
|   └── yoon_EDA.ipynb
|
├──📁 config
|   └── config_baseline.yaml
|
├──📂 etc 
|   └── catboost.ipynb
|
├──📂 src
|   ├── 📂 data
|   │    ├── __init__.py 
|   │    ├── basic_data.py 
|   │    ├── context_data.py 
|   │    ├── image_data.py 
|   │    └── text_data.py
|   |
|   ├── 📂 ensemble
|   │    └── ensembles.py
|   |
|   ├── 📂 loss
|   │    ├── __init__.py
|   │    └── loss.py
|   |
|   ├── 📂 models
|   │    ├── .DS_Store
|   │    ├── __init__.py 
|   │    ├── _helper.py 
|   │    ├── image_FM.py 
|   │    └── text_FM.py
|   |
|   ├── 📂 train
|   │    ├── __init__.py 
|   │    └── trainer.py
|   ├── __init__.py
|   ├── utils.py
|   └── README.md
├── .gitignore
├── ensemble.py
├── main.py
├── optuna_study.py
├── requirement.txt
├── run_baseline.sh
└── README.md
```

<br></br>

## ▶️ 실행 방법 ▶️

- Package install
    
    ```bash
    pip install -r requirements.txt
    ```
    
- Model training
    
    ```bash
    # main.py 실행
    python main.py  -c config/config_baseline.yaml  -m Image_DeepFM  -w True  -r Image_DeepFM_baseline
    python main.py  -c config/config_baseline.yaml  -m Text_DeepFM  -w True  -r Text_DeepFM_baseline
    python main.py  -c config/config_baseline.yaml  -m CatBoost  -w True  -r CatBoost
    
    # optuna_study.py 실행(트리 모델만 가능)
    python optuna_study.py  -c config/config_baseline.yaml  -m XGBoost  -w True  -r XGBoost
    
    # ensemble.py 실행
    python ensemble.py --ensemble_model 'XGBoost','CatBoost' --ensemble_strategy weighted --ensemble_weight 6,4
    ```

<br></br>

## 🥇 Result 🥇
#### 제출 1 - CatBoost 단일 모델

Stratified K-Fold와 optuna를 적용하여 CatBoost를 훈련시켰습니다.
![image](https://github.com/user-attachments/assets/4c1dc7f8-c01b-4004-b3a3-f03ffdcd6136)
→ 1등!
![image](https://github.com/user-attachments/assets/d599a549-7d05-4172-8a10-99c431920eac)


#### 제출 2 - CatBoost + Image DeepFM + Text DeepFM
CatBoost와 베이스라인 코드로 주어진 Image DeepFM, Text DeepFM을 각각 8 : 1 : 1의 비율로 하여 소프트 보팅을 적용하였습니다.
![image](https://github.com/user-attachments/assets/6cd0506e-38f6-4093-94a7-35a3b65d675d)



<br></br>
## GitHub Convention

- ***main*** branch는 배포이력을 관리하기 위해 사용,

  ***house*** branch는 기능 개발을 위한 branch들을 병합(merge)하기 위해 사용
- 모든 기능이 추가되고 버그가 수정되어 배포 가능한 안정적인 상태라면 *house* branch에 병합(merge)
- 작업을 할 때에는 개인의 branch를 통해 작업
- EDA
    
    branch명 형식은 “**EDA-자기이름**” 으로 작성 ex) EDA-TaroSin
    
    파일명 형식은 “**name_EDA**” 으로 작성 ex) TaroSin_EDA
    
- 데이터 전처리팀 branch 관리 규칙
    
    ```
    book 
    └── data
    ```
    
- 모델팀 branch 관리 규칙
    
    ```
    book 
    └── model
        ├── model-modularization   # model 개발 및 모듈화 작업
        ├── model-stratifiedkfold  # stratifiedkfold 로직 개발
        ├── model-optuna           # optuna 로직 개발
        └── model-experiment       # 모델 실험
    ```
    
- *master(main)* Branch에 Pull request를 하는 것이 아닌,
    
    ***data*** Branch 또는 ***model*** Branch에 Pull request 요청
    
- commit message는 아래와 같이 구분해서 작성 (한글)

  ex) git commit -m “**docs**: {내용} 문서 작성”
  
  ex) git commit -m “**feat**: {내용} 추가”
  
  ex) git commit -m “**fix**: {내용} 수정”
  
  ex) git commit -m “**test**: {내용} 테스트”

- pull request merge 담당자 : **data - 근서** / **model - 윤경** / **최종 - 영균**

  나머지는 ***house*** branch 건드리지 말 것!

  merge commit message는 아래와 같이 작성

  ex) “**merge**: {내용} 병합“
- **Issues**, **Pull request**는 Template에 맞추어 작성 (커스텀 Labels 사용)
Issues → 작업 → PR 순으로 진행

<br></br>

## Code Convention

- 문자열을 처리할 때는 작은 따옴표를 사용하도록 합니다.
- 클래스명은 `카멜케이스(CamelCase)` 로 작성합니다. </br>
  함수명, 변수명은 `스네이크케이스(snake_case)`로 작성합니다.
- 객체의 이름은 해당 객체의 기능을 잘 설명하는 것으로 정합니다.  
    ```python
    # bad
    a = ~~~
    # good
    lgbm_pred_y = ~~~
    ```
- 가독성을 위해 한 줄에 하나의 문장만 작성합니다.
- 들여쓰기는 4 Space 대신 Tab을 사용합시다.
- 주석은 설명하려는 구문에 맞춰 들여쓰기, 코드 위에 작성 합니다.
    ```python
    # good
    def some_function():
      ...
    
      # statement에 관한 주석
      statements
    ```
    
- 대구분 주석은 ###으로 한 줄 위에 작성 합니다.
    
    ```python
    # good
    ### normalization
    
    def standardize_feature(
    ```
    
- 키워드 인수를 나타낼 때나 주석이 없는 함수 매개변수의 기본값을 나타낼 때 기호 주위에 공백을 사용하지 마세요.
    
    ```python
    # bad
    def complex(real, imag = 0.0):
        return magic(r = real, i = imag)
    # good
    def complex(real, imag=0.0):
        return magic(r=real, i=imag)
    ```
    
- 연산자 사이에는 공백을 추가하여 가독성을 높입니다.
    
    ```python
    a+b+c+d # bad
    a + b + c + d # good
    ```
    
- 콤마(,) 다음에 값이 올 경우 공백을 추가하여 가독성을 높입니다.
    
    ```python
    arr = [1,2,3,4] # bad
    arr = [1, 2, 3, 4] # good
    ```
