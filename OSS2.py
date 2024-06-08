# project2.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# 1단계: 데이터 전처리 및 클러스터링

# 데이터 불러오기
file_path = 'ratings.dat'  # 파일 경로
column_names = ['UserID', 'ItemID', 'Rating', 'Timestamp']  # 데이터 파일의 컬럼 이름 정의
data = pd.read_csv(file_path, sep='::', names=column_names, engine='python')  # 데이터를 데이터프레임으로 읽어옴

# User x Item 매트릭스 생성
user_item_matrix = data.pivot(index='UserID', columns='ItemID', values='Rating').fillna(
    0)  # 피벗 테이블을 생성하여 User x Item 매트릭스 생성, 결측값은 0으로 채움
user_item_matrix_np = user_item_matrix.to_numpy()  # 매트릭스를 numpy 배열로 변환

# K-Means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=0)  # KMeans 객체 생성, 클러스터 개수는 3개
clusters = kmeans.fit_predict(user_item_matrix_np)  # K-Means 클러스터링을 수행하고 클러스터 라벨 반환

# 각 사용자의 클러스터 할당
user_clusters = pd.DataFrame(
    {'UserID': user_item_matrix.index, 'Cluster': clusters})  # 각 사용자에 대해 클러스터 할당 결과를 데이터프레임으로 저장

# 2단계: 그룹 추천 알고리즘 구현

# 그룹별 사용자 데이터 가져오기
grouped_ratings = []  # 그룹별 사용자 평점을 저장할 리스트
for cluster in range(3):  # 클러스터 0, 1, 2에 대해 반복
    user_ids = user_clusters[user_clusters['Cluster'] == cluster]['UserID']  # 현재 클러스터에 속한 사용자 ID를 가져옴
    group_ratings = user_item_matrix.loc[user_ids]  # 현재 클러스터에 속한 사용자의 평점 데이터를 추출
    grouped_ratings.append(group_ratings)  # 추출된 평점 데이터를 리스트에 추가


# Additive Utilitarian (AU) 알고리즘
def additive_utilitarian(group_ratings):
    return group_ratings.sum(axis=0)  # 그룹 내 모든 사용자의 평점을 더함


# Average (Avg) 알고리즘
def average(group_ratings):
    return group_ratings.mean(axis=0)  # 그룹 내 모든 사용자의 평점 평균을 계산


# Simple Count (SC) 알고리즘
def simple_count(group_ratings):
    return (group_ratings > 0).sum(axis=0)  # 평점을 부여한 항목의 수를 카운트


# Approval Voting (AV) 알고리즘
def approval_voting(group_ratings, threshold=4):
    return (group_ratings >= threshold).sum(axis=0)  # 지정된 임계값을 초과하는 평점을 카운트


# Borda Count (BC) 알고리즘
def borda_count(group_ratings):
    ranks = group_ratings.rank(axis=1, method='average', ascending=False)  # 각 사용자의 평점을 순위로 변환
    return ranks.sum(axis=0)  # 순위 점수를 합산


# Copeland Rule (CR) 알고리즘 (벡터화된 버전)
def copeland_rule(group_ratings):
    n_items = group_ratings.shape[1]
    win_matrix = np.zeros((n_items, n_items))  # 승패 매트릭스 초기화

    for i in range(n_items):
        win_matrix[i] = (group_ratings.iloc[:, i].values[:, None] > group_ratings.values).sum(axis=0)  # i 아이템의 승리 횟수

    loss_matrix = win_matrix.T  # 패배 매트릭스는 승리 매트릭스의 전치 행렬
    copeland_scores = win_matrix.sum(axis=1) - loss_matrix.sum(axis=1)  # Copeland 점수 계산

    return copeland_scores


# 결과를 파일로 저장하는 함수
def save_recommendations_to_file(group_number, recommendations):
    with open(f'group_{group_number}_recommendations.txt', 'w') as f:
        for algorithm, items in recommendations.items():
            f.write(f"{algorithm} Top 10 Items: {items}\n")


# 각 그룹에 대해 알고리즘 적용 및 추천 결과 저장
for i, group_ratings in enumerate(grouped_ratings):  # 각 그룹에 대해 반복
    recommendations = {}  # 추천 결과를 저장할 딕셔너리
    print(f"\nGroup {i + 1} Recommendations:")  # 그룹 번호 출력
    algorithms = {
        'Additive Utilitarian': additive_utilitarian,
        'Average': average,
        'Simple Count': simple_count,
        'Approval Voting': lambda x: approval_voting(x, threshold=4),
        'Borda Count': borda_count,
        'Copeland Rule': copeland_rule  # Copeland Rule 사용
    }

    for name, algorithm in algorithms.items():  # 각 알고리즘에 대해 반복
        scores = algorithm(group_ratings)  # 알고리즘을 적용하여 아이템 점수 계산
        top_items = np.argsort(scores)[-10:][::-1]  # 상위 10개 아이템을 내림차순으로 정렬하여 선택
        recommendations[name] = top_items.tolist()  # 추천 결과를 딕셔너리에 저장
        print(f"{name} Top 10 Items: {top_items.tolist()}")  # 알고리즘 이름과 상위 10개 아이템 출력

    # 각 그룹의 추천 결과를 파일로 저장
    save_recommendations_to_file(i + 1, recommendations)
