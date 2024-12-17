
# **Deeplearning_Study**


## **Week 1: Image Classification**

- **목적**: MNIST 및 CIFAR-10 데이터셋을 활용하여 이미지 분류(Image Classification)를 수행.
 
- **사용된 모델**:
  - ResNet-18
  - VGG-19
    
- **내용**:
  - 사전 학습된 모델(Pre-trained model)을 사용하여 MNIST 및 CIFAR-10 데이터셋에서 이미지 분류를 수행.
  - 모델의 학습 및 평가를 통해 기본적인 딥러닝 기반 이미지 분류 기법을 이해


## **Week 2: Image Classification with Data Augmentation**

- **목적**: 1주차의 내용을 확장하여 ResNet-18과 VGG-19 모델을 직접 구현하고, 성능을 개선.

- **사용된 기법**:
  - 데이터 증강(Data Augmentation)
  - 그래디언트 클리핑(Gradient Clipping)
    
- **내용**:
  - ResNet-18과 VGG-19 모델을 처음부터 직접 구현.
  - 데이터 증강 기법(예: Crop, , 스케일 변환 등)을 적용하여 데이터 다양성을 확보.
  - 그래디언트 클리핑을 도입하여 학습 안정성과 성능을 개선.
  - MNIST와 CIFAR-10 데이터셋에서 더 나은 분류 성능을 달성.



## **Week 5: Text Classification**

- **목적**: IMDB 데이터셋을 활용하여 텍스트 분류(Text Classification)를 수행.
 
- **사용된 모델**:
  - Transformer Encoder
  - BERT (Bidirectional Encoder Representations from Transformers)
  - LSTM (Long Short-Term Memory)
   
- **내용**:
  - IMDB 데이터셋에서 긍정(Positive)과 부정(Negative) 리뷰를 분류하는 작업 수행.
  - 다양한 모델 아키텍처를 비교하고, 각 모델의 성능과 특징 분석.
  - Transformer Encoder, BERT, LSTM 모델의 차이를 이해하고 실험.


## **Week 7: Transfomrer Implementation** 

- **목적**: Text classification이 아닌 machine translation처럼 문장을 생성할 수 있는 모델 구현

- **사용된 데이터**:
  - WMT2016 Multi-modal 데이터셋
  - Custom Dataset을 활용하여 데이터 로드

- **사용된 모델**:
  - 직접 구현한 Transformer 모델
  - Hugging Face 라이브러리 활용 (pretrained되지 않은 모델 불러와 학습 진행)
    
- **목표**:
  - Test 데이터셋에 대한 BLEU-4 Score 0.15 이상 달성


## **Week 9: P_Transfomrer** 

- **목적**: ransformer 모델을 원하는 방식으로 변형하여 사용할 수 있도록 구현 연습.
 
- **사용된 모델**:
  - 기존 Original Transformer를 P-Transformer로 변형
     
- **목표**:
  - Test 데이터셋에 대한 BLEU-4 Score 0.15 이상 달성

  
