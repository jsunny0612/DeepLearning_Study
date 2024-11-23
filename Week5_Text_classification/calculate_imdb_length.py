import pandas as pd
import matplotlib.pyplot as plt

# 데이터셋 경로
train_file_path = "/media/user/HDD2/sh/Chung_Ang/Week5_Text_classification/data/IMDb_movie_reviews/IMDB_train.csv"
val_file_path = "/media/user/HDD2/sh/Chung_Ang/Week5_Text_classification/data/IMDb_movie_reviews/IMDB_validation.csv"
test_file_path = "/media/user/HDD2/sh/Chung_Ang/Week5_Text_classification/data/IMDb_movie_reviews/IMDB_test.csv"

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)
validation_data = pd.read_csv(val_file_path)

# 텍스트 길이 계산 함수
def calculate_imdb_length(data):
    # 각 텍스트의 길이를 리스트로 반환
    lengths = [len(text.split()) for text in data['text'].values]
    return lengths

# 각각의 데이터셋에 대해 텍스트 길이를 계산
train_lengths = calculate_imdb_length(train_data)
test_lengths = calculate_imdb_length(test_data)
validation_lengths = calculate_imdb_length(validation_data)

# 각 데이터셋의 텍스트 길이 시각화 함수
def draw_graph(lengths, dataset_name):
    plt.hist(lengths, bins=50)
    plt.title(f'{dataset_name}')
    plt.xlabel('length')
    plt.ylabel('frequency')
    plt.show()

# Train, Test, Validation 길이 분포를 각각 출력
draw_graph(train_lengths, 'Train')
draw_graph(test_lengths, 'Test')
draw_graph(validation_lengths, 'Validation')

# 평균 계산
print(f'Train data 평균 길이: {sum(train_lengths)/len(train_lengths)}')
print(f'Test data 평균 길이: {sum(test_lengths)/len(test_lengths)}')
print(f'Validation data 평균 길이: {sum(validation_lengths)/len(validation_lengths)}')
