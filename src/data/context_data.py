import numpy as np
import pandas as pd
import regex
import torch
from torch.utils.data import TensorDataset, DataLoader
from .basic_data import basic_data_split

def str2list(x: str) -> list:
    '''문자열을 리스트로 변환하는 함수'''
    return x[2:-2].split(', ')


def split_location(x: str) -> list:
    '''
    Parameters
    ----------
    x : str
        location 데이터

    Returns
    -------
    res : list
        location 데이터를 나눈 뒤, 정제한 결과를 반환합니다.
        순서는 country, state, city, ... 입니다.
    '''
    res = x.split(',')
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r'[^a-zA-Z/ ]', '', i) for i in res]  # remove special characters
    res = [i if i not in ['n/a', ''] else np.nan for i in res]  # change 'n/a' into NaN
    res.reverse()  # reverse the list to get country, state, city, ... order

    for i in range(len(res)-1, 0, -1):
        if (res[i] in res[:i]) and (not pd.isna(res[i])):  # remove duplicated values if not NaN
            res.pop(i)

    return res

def text_preprocessing(summary: str) -> str:
    """
    주어진 텍스트 요약을 전처리합니다.

    1. 특수 문자 제거
    2. 알파벳과 숫자, 공백을 제외한 모든 문자 제거
    3. 여러 개의 공백을 하나의 공백으로
    4. 문자열의 앞뒤 공백 제거
    5. 모든 문자를 소문자로 변환

    Args:
        summary (str): 전처리할 텍스트 문자열

    Returns:
        str: 전처리된 텍스트 문자열. 입력이 NaN인 경우 "unknown" 반환.
    """
    if pd.isna(summary):
        return 'unknown'  # NaN일 경우 "unknown" 반환
    
    summary = regex.sub('[.,\'\"''\"!?]', '', summary)  # 특수 문자 제거
    summary = regex.sub('[^0-9a-zA-Z\s]', '', summary)  # 알파벳과 숫자, 공백 제외한 문자 제거
    summary = regex.sub('\s+', ' ', summary)  # 여러 개의 공백을 하나의 공백으로
    summary = summary.lower()  # 소문자로 변환
    summary = summary.strip()  # 앞뒤 공백 제거
    return summary

def categorize_publication(x: int, a: int) -> int:
    """
    주어진 연도를 특정 기준에 따라 카테고리화하는 함수입니다.

    Parameters
    ----------
    x : int
        책의 발행 연도.
    a : int
        연도를 그룹화할 때 사용할 기준값 (예: 5년 단위로 그룹화).

    Returns
    -------
    int
        카테고리화된 연도를 반환합니다. 
        - 1970년 이하의 연도는 1970으로 반환합니다.
        - 2000년 초과의 연도는 2006으로 반환합니다.
        - 나머지 연도는 a 값에 맞게 그룹화하여 반환합니다.

    Example
    -------
    books['years'] = books['year_of_publication'].apply(lambda x: categorize_publication(x, 5))
    print(books['years'].value_counts())
    """
    if x <= 1970:
        return 1970
    elif x > 2000:
        return 2006
    else:
        return x // a * a

def extract_language_from_isbn(isbn):
    """
    ISBN 정보를 사용하여 언어 코드를 추출하는 함수입니다.

    Parameters
    ----------
    isbn : str
        책의 ISBN 번호.

    Returns
    -------
    str
        ISBN에서 추출한 언어 코드. ISBN이 비어있거나 형식에 맞지 않을 경우 최빈값 'en'을 반환합니다.
        - isbn_language_map 참고
        - 기타 언어 코드: isbn_language_map에 정의된 국가 코드를 기반으로 반환
    """
    isbn_language_map = {
        '0': 'en', '1': 'en', '2': 'fr', '3': 'de', '4': 'ja',
        '5': 'ru', '7': 'zh-CN', '82': 'no', '84': 'es', '87': 'da',
        '88': 'it', '89': 'ko', '94': 'nl', '600': 'fa', '602': 'ms',
        '606': 'ro', '604': 'vi', '618': 'el', '967': 'ms', '974': 'th',
        '989': 'pt'
    }
    # isbn_language_map = {
    #     '0': 'en', '1': 'en', '2': 'fr', '3': 'de', '4': 'ja', '5': 'ru', '7': 'zh-CN',
    #     '82': 'no', '84': 'es', '87': 'da', '88': 'it', 
    #     '602': 'ms', '967': 'ms', '974': 'th'
    #     #'89': 'ko', '94': 'nl', # '600': 'fa', '604': 'vi', '606': 'ro', '618': 'el', '989': 'pt'
    # }
    if not isbn or not isbn.isdigit():
        return 'en'  # 기본값 영어권
    for prefix, language in isbn_language_map.items():
        if isbn.startswith(prefix):
            return language
    return 'en'  # 기본값 영어권

def replace_language_using_isbn(books):
    """
    ISBN 정보를 활용하여 language 결측치를 대체하는 함수입니다.

    Parameters
    ----------
    books : pd.DataFrame
        책 정보가 담긴 DataFrame. 반드시 'isbn' 및 'language' 열을 포함해야 합니다.

    Returns
    -------
    pd.DataFrame
        language 결측치가 ISBN 정보를 사용해 대체된 DataFrame. ISBN에서 언어를 추출할 수 없는 경우
        기본값 'en'으로 대체됩니다.

    Example
    -------
    books = replace_language_using_isbn(books)
    """
    books['extracted_language'] = books['isbn'].apply(extract_language_from_isbn)
    books['language'] = books.apply(
        lambda row: row['extracted_language'] if pd.isna(row['language']) else row['language'],
        axis=1
    )
    books.drop(columns=['extracted_language'], inplace=True)
    return books
    
def categorize_age(x: int, a: int) -> int:
    """
    주어진 나이를 특정 기준에 따라 카테고리화하는 함수입니다.

    Parameters
    ----------
    x : int
        유저의 나이.
    a : int
        나이를 그룹화할 때 사용할 기준값 (예: 10년 단위로 그룹화).

    Returns
    -------
    int
        카테고리화된 나이를 반환합니다. 
        - 20년 미만의 나이는 10으로 반환합니다.
        - 60년 이상의 나이는 60으로 반환합니다.
        - 나머지 나이는 a 값에 맞게 그룹화하여 반환합니다.
    """
    if x < 20:
        return 10
    elif x >= 60:
        return 60
    else:
        return x // a * a

def process_context_data(users, books):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    
    Returns
    -------
    label_to_idx : dict
        데이터를 인덱싱한 정보를 담은 딕셔너리
    idx_to_label : dict
        인덱스를 다시 원래 데이터로 변환하는 정보를 담은 딕셔너리
    train_df : pd.DataFrame
        train 데이터
    test_df : pd.DataFrame
        test 데이터
    """

    users_ = users.copy()
    books_ = books.copy()

    # 데이터 전처리

    ##################### books
    books_['book_title'] = books_['book_title'].apply(text_preprocessing)
    books_['book_author'] = books_['book_author'].apply(text_preprocessing)
    books_['publisher'] = books_['publisher'].apply(text_preprocessing)
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: categorize_publication(x, 5))
    books_ = replace_language_using_isbn(books_)
    books_['category'] = books_['category'].apply(lambda x: str2list(x)[0] if not pd.isna(x) else np.nan)
    books_['category'] = books_['category'].apply(text_preprocessing)
    high_categories = ['fiction', 'biography', 'history', 'religion', 'nonfiction', 'social', 'science', 'humor', 'body', 
                   'business', 'economics', 'cook', 'health', 'fitness', 'famil', 'relationship', 
                   'computer', 'travel', 'selfhelp', 'psychology', 'poetry', 'art', 'critic', 'nature', 'philosophy', 
                   'reference','drama', 'sports', 'politic', 'comic', 'novel', 'craft', 'language', 'education', 'crime', 'music', 'pet', 
                   'child', 'collection', 'mystery', 'garden', 'medical', 'author', 'house','technology', 'engineering', 'animal', 'photography',
                   'adventure', 'game', 'science fiction', 'architecture', 'law', 'fantasy', 'antique', 'friend', 'brother', 'sister', 'cat',
                   'math', 'christ', 'bible', 'fairy', 'horror', 'design', 'adolescence', 'actor', 'dog', 'transportation', 'murder', 'adultery', 'short', 'bear'
                   ]
    # high_category 열을 초기화
    books_['high_category'] = None
    # 각 카테고리에 대해 반복하며 매핑
    for high_category in high_categories:
        # category 열에서 high_category가 포함된 행을 찾고, 해당 행의 high_category 열을 업데이트
        books_.loc[books_['category'].str.contains(high_category, case=False, na=False), 'high_category'] = high_category
    books_['high_category'] = books_['high_category'].fillna('others') # 결측치를 'others'로 대체

    ##################### users
    users_['age'] = users_['age'].fillna(users_['age'].mean())
    users_['age_range'] = users_['age'].apply(lambda x: categorize_age(x, 10))

    users_['location_list'] = users_['location'].apply(lambda x: split_location(x)) 
    users_['location_country'] = users_['location_list'].apply(lambda x: x[0])
    users_['location_state'] = users_['location_list'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    users_['location_city'] = users_['location_list'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
    for idx, row in users_.iterrows():
        if (not pd.isna(row['location_state'])) and pd.isna(row['location_country']):
            fill_country = users_[users_['location_state'] == row['location_state']]['location_country'].mode()
            fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
            users_.loc[idx, 'location_country'] = fill_country
        elif (not pd.isna(row['location_city'])) and pd.isna(row['location_state']):
            if not pd.isna(row['location_country']):
                fill_state = users_[(users_['location_country'] == row['location_country']) 
                                    & (users_['location_city'] == row['location_city'])]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                users_.loc[idx, 'location_state'] = fill_state
            else:
                fill_state = users_[users_['location_city'] == row['location_city']]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                fill_country = users_[users_['location_city'] == row['location_city']]['location_country'].mode()
                fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
                users_.loc[idx, 'location_country'] = fill_country
                users_.loc[idx, 'location_state'] = fill_state
    users_['location_country'] = users_['location_country'].fillna(users_['location_country'].mode()[0])

    return users_, books_


def context_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    
    Returns
    -------
    data : dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다.
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    users_, books_ = process_context_data(users, books)
    
    
    # 유저 및 책 정보를 합쳐서 데이터 프레임 생성
    # 사용할 컬럼을 user_features와 book_features에 정의합니다. (단, 모두 범주형 데이터로 가정)
    # 베이스라인에서는 가능한 모든 컬럼을 사용하도록 구성하였습니다.
    # NCF를 사용할 경우, idx 0, 1은 각각 user_id, isbn이어야 합니다.
    user_features = ['user_id', 'age_range', 'location_country']
    book_features = ['isbn', 'book_title', 'book_author', 'publisher', 'language', 'high_category', 'publication_range']
    sparse_cols = ['user_id', 'isbn'] + list(set(user_features + book_features) - {'user_id', 'isbn'}) if args.model == 'NCF' \
                   else user_features + book_features

    # 선택한 컬럼만 추출하여 데이터 조인
    train_df = train.merge(users_, on='user_id', how='left')\
                    .merge(books_, on='isbn', how='left')[sparse_cols + ['rating']]
    test_df = test.merge(users_, on='user_id', how='left')\
                  .merge(books_, on='isbn', how='left')[sparse_cols]
    all_df = pd.concat([train_df, test_df], axis=0)

    # feature engineering
    user_id_counts = train_df['user_id'].value_counts()
    train_df['user_review_counts'] = train_df['user_id'].map(user_id_counts)
    test_df['user_review_counts'] = test_df['user_id'].map(user_id_counts)
    test_df['user_review_counts'] = test_df['user_review_counts'].fillna(0)

    book_isbn_counts = train_df['isbn'].value_counts()
    train_df['book_review_counts'] = train_df['isbn'].map(book_isbn_counts)
    test_df['book_review_counts'] = test_df['isbn'].map(book_isbn_counts)
    test_df['book_review_counts'] = test_df['book_review_counts'].fillna(0)

    # feature_cols의 데이터만 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = pd.Categorical(train_df[col], categories=unique_labels).codes
        test_df[col] = pd.Categorical(test_df[col], categories=unique_labels).codes
        # train_df[col] = train_df[col].map(label2idx[col])
        # test_df[col] = test_df[col].map(label2idx[col])

    # 수치형 변수의 경우 label2idx에 추가 (차원 수는 1로 설정)
    label2idx['user_review_counts'] = {0: 0}
    idx2label['user_review_counts'] = {0: 'user_review_counts'}
    label2idx['book_review_counts'] = {0: 0}
    idx2label['book_review_counts'] = {0: 'book_review_counts'}
    field_dims = [len(label2idx[col]) for col in train_df.columns if col != 'rating']
    sparse_cols += ['user_review_counts', 'book_review_counts'] # 추가한 필드명도 포함

    data = {
            'train':train_df,
            'test':test_df,
            'field_names':sparse_cols,
            'field_dims':field_dims,
            'label2idx':label2idx,
            'idx2label':idx2label,
            'sub':sub,
            }

    return data


def context_data_split(args, data):
    '''data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다.'''
    return basic_data_split(args, data)


def context_data_loader(args, data):
    """
    Parameters
    ----------
    args.dataloader.batch_size : int
        데이터 batch에 사용할 데이터 사이즈
    args.dataloader.shuffle : bool
        data shuffle 여부
    args.dataloader.num_workers: int
        dataloader에서 사용할 멀티프로세서 수
    args.dataset.valid_ratio : float
        Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용합니다.
    data : dict
        context_data_load 함수에서 반환된 데이터
    
    Returns
    -------
    data : dict
        DataLoader가 추가된 데이터를 반환합니다.
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values)) if args.dataset.valid_ratio != 0 else None
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
