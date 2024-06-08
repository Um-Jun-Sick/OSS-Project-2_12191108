import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# 1단계: 데이터 전처리 및 클러스터링

# 데이터 불러오기
data_file_path = 'ratings.dat'  # 파일 경로
data_columns = ['User_ID', 'Item_ID', 'User_Rating', 'Rate_Time']  # 컬럼 이름 정의
ratings_df = pd.read_csv(data_file_path, sep='::', names=data_columns, engine='python')  # 데이터를 데이터프레임으로 읽어옴

# 사용자-아이템 매트릭스 생성
user_item_df = ratings_df.pivot(index='User_ID', columns='Item_ID', values='User_Rating').fillna(
    0)  # 피벗 테이블을 생성하여 사용자-아이템 매트릭스 생성, 결측값은 0으로 채움
user_item_matrix = user_item_df.to_numpy()  # 매트릭스를 numpy 배열로 변환

# K-Means 클러스터링 수행
custom_kmeans = KMeans(n_clusters=3, random_state=0)  # KMeans 객체 생성, 클러스터 개수는 3개
cluster_labels = custom_kmeans.fit_predict(user_item_matrix)  # K-Means 클러스터링을 수행하고 클러스터 라벨 반환

# 사용자에게 클러스터 할당
user_cluster_df = pd.DataFrame(
    {'User_ID': user_item_df.index, 'User_Cluster': cluster_labels})  # 각 사용자에 대해 클러스터 할당 결과를 데이터프레임으로 저장

# 2단계: 그룹 추천 알고리즘 구현

# 그룹별 사용자 데이터 가져오기
grouped_user_ratings_list = []  # 그룹별 사용자 평점을 저장할 리스트
for cluster_idx in range(3):  # 클러스터 0, 1, 2에 대해 반복
    cluster_user_ids = user_cluster_df[user_cluster_df['User_Cluster'] == cluster_idx]['User_ID']  # 현재 클러스터에 속한 사용자 ID를 가져옴
    cluster_ratings_df = user_item_df.loc[cluster_user_ids]  # 현재 클러스터에 속한 사용자의 평점 데이터를 추출
    grouped_user_ratings_list.append(cluster_ratings_df)  # 추출된 평점 데이터를 리스트에 추가


# Sum_Ratings 알고리즘
def sum_ratings(ratings_group):
    return ratings_group.sum(axis=0)  # 그룹 내 모든 사용자의 평점을 더함


# Mean_Ratings 알고리즘
def mean_ratings(ratings_group):
    return ratings_group.mean(axis=0)  # 그룹 내 모든 사용자의 평점 평균을 계산


# Count_Rated_Items 알고리즘
def count_rated_items(ratings_group):
    return (ratings_group > 0).sum(axis=0)  # 평점을 부여한 항목의 수를 카운트


# Threshold_Voting 알고리즘
def threshold_voting(ratings_group, threshold=4):
    return (ratings_group >= threshold).sum(axis=0)  # 지정된 임계값을 초과하는 평점을 카운트


# Rank_Sum 알고리즘
def rank_sum(ratings_group):
    ranking = ratings_group.rank(axis=1, method='average', ascending=False)  # 각 사용자의 평점을 순위로 변환
    return ranking.sum(axis=0)  # 순위 점수를 합산


# Copeland_Algorithm 알고리즘 (벡터화된 버전)
def copeland_algorithm(ratings_group):
    num_items = ratings_group.shape[1]
    win_matrix = np.zeros((num_items, num_items))  # 승패 매트릭스 초기화

    for item_idx in range(num_items):
        win_matrix[item_idx] = (ratings_group.iloc[:, item_idx].values[:, None] > ratings_group.values).sum(axis=0)  # i 아이템의 승리 횟수

    loss_matrix = win_matrix.T  # 패배 매트릭스는 승리 매트릭스의 전치 행렬
    copeland_scores = win_matrix.sum(axis=1) - loss_matrix.sum(axis=1)  # Copeland 점수 계산

    return copeland_scores


# 결과를 파일로 저장하는 함수
def save_recommendations(group_num, recommendations):
    with open(f'group_{group_num}_recommendations.txt', 'w') as file:
        for algo_name, items in recommendations.items():
            file.write(f"{algo_name} Top 10 Items: {items}\n")


# 각 그룹에 대해 알고리즘 적용 및 추천 결과 저장
for idx, ratings in enumerate(grouped_user_ratings_list):  # 각 그룹에 대해 반복
    recommendation_results = {}  # 추천 결과를 저장할 딕셔너리
    print(f"\nGroup {idx + 1} Recommendations:")  # 그룹 번호 출력
    recommendation_algorithms = {
        'Sum_Ratings': sum_ratings,
        'Mean_Ratings': mean_ratings,
        'Count_Rated_Items': count_rated_items,
        'Threshold_Voting': lambda x: threshold_voting(x, threshold=4),
        'Rank_Sum': rank_sum,
        'Copeland_Algorithm': copeland_algorithm  # Copeland Rule 사용
    }

    for algo_name, algo_func in recommendation_algorithms.items():  # 각 알고리즘에 대해 반복
        item_scores = algo_func(ratings)  # 알고리즘을 적용하여 아이템 점수 계산
        top_items = np.argsort(item_scores)[-10:][::-1]  # 상위 10개 아이템을 내림차순으로 정렬하여 선택
        recommendation_results[algo_name] = top_items.tolist()  # 추천 결과를 딕셔너리에 저장
        print(f"{algo_name} Top 10 Items: {top_items.tolist()}")  # 알고리즘 이름과 상위 10개 아이템 출력

    # 각 그룹의 추천 결과를 파일로 저장
    save_recommendations(idx + 1, recommendation_results)
