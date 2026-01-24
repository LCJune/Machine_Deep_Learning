## 시퀀스 투 시퀀스(Sequence To Sequence)
시퀀스 데이터를 입력받아 다시 시퀀스 데이터를 출력하는 작업으로, 텍스트 요약이나 번역 등의 작업이 이에 속한다.  
두 개의 신경망을 사용한 인코더-디코더 구조가 널리 쓰인다.  

## 어텐션 메커니즘(Attention Machanism) 
1. 어텐션 메커니즘의 핵심 개념  
인코더-디코더 구조에 사용된 순환 신경망의 성능을 향상시키기 위해 고안됐다.  
모델이 현재 예측을 수행하는 시점에서, 입력 시퀀스의 모든 위치 중 ‘어디에 얼마나 집중할 것인지’를 동적으로 결정하는 계산 구조이다.  
디코더가 모든 타임 스텝에서 인코더가 출력한 은닉 상태를 참조하며, 이를 통해 어느 타임스텝의 토큰에 집중할지 가중치를 학습한다.      
<br />

어텐션 메커니즘에서 각 토큰은 세 가지 벡터로 변환된다.  
* Query(Q): "무엇을 찾고 있는가" (비교 기준) -> 질문자
* Key(K): "각 토큰이 어떤 정보를 갖고 있는가"  (비교 대상의 설명) -> 주변 단어들의 정보
* Value(V): "실제로 전달할 정보"  (전달 정보) -> 실제 정보값

2. 쿼리, 키, 밸류의 생성  
> <img width="492" height="295" alt="image" src="https://github.com/user-attachments/assets/d51dfc56-be52-458a-b7a1-fc3ae684370b" />  
n: 입력 시퀀스 길이, d_model: 모델 차원  
<br />

3. 어텐션 스코어(유사도 계산) 생성  
> <img width="578" height="164" alt="image" src="https://github.com/user-attachments/assets/b8cff54e-6cfc-491c-b875-c5c2d4749430" />  
<br />

4. Scaled-Dot Product  
> <img width="687" height="99" alt="image" src="https://github.com/user-attachments/assets/4ae6d133-b631-4159-9bae-c502bbcbcadb" />     
<br />

5. 어텐션 가중치(Attention Weights)
> <img width="506" height="161" alt="image" src="https://github.com/user-attachments/assets/98647ea7-f2b6-4abd-b669-a2d4fff0890f" />   
중요도(score)를 확률분포로 정규화 한다.
<br />

6. Value의 가중합(최종출력)
>  <img width="485" height="200" alt="image" src="https://github.com/user-attachments/assets/d8dac919-7cf0-4754-9162-c3ab28fc9a26" />   
중요한 토큰의 정보를 더 많이 반영하도록 한다.  
<br />

7. 전체 어텐션 수식 정리
> <img width="360" height="55" alt="image" src="https://github.com/user-attachments/assets/daf17645-7647-4393-a213-08e5442d90e3" />  
<br />

어텐션 메커니즘은 기존의 RNN이 가지고 있던 장거리 의존성 문제, 병렬 처리의 어려움, 고정 크기 문맥 벡터의 한계 등의 단점을 해결하고자 개발되었다.  
이러한 어텐션 메커니즘은 긴 텍스트를 처리할 때 정보 손실 데 높은 성능을 보인다.
그러나 어텐션 가중치 계산을 위해 모든 타임스텝의 은닉 상태를 저장하므로, 메모리 사용량과 연산량이 높다.  
<br />

## 트랜스포머(Transformer)
인코더 - 디코더 구조를 유지하고 순환 신경망을 완전히 제거하는 대신 셀프 어텐션 메커니즘을 기반으로 텍스트를 처리하는 모델이다.  
현재 NLP(Natural Language Processing) 분야의 핵심 기술이다.  
트랜스포머의 핵십 구성 요소는 다음과 같다.  
1. 인코더 블록
2. 디코더 블록
3. 잔차 연결 & 층 정규화


### 셀프 어텐션 메커니즘(Self-Attention Machanism)   
> <img width="1498" height="1092" alt="image" src="https://github.com/user-attachments/assets/5f1e5d04-e112-4292-a7c9-131c09442b79" />

문장 내에서 각 단어가 다른 단어들과 어떤 관게인지 파악하고, 중요도를 계산할 수 있도록 하는 메커니즘이다.  
인코더에 입력되는 토큰만을 사용하여 어텐션 가중치를 계산한다.    
<br />  

위 그림과 같이 셀프 어텐션 연산을 수행하는 하나의 단위를 **어텐션 헤드(Attention Head)** 라고 한다.  
트랜스포머는 여러개의 어텐션 헤드를 사용하며, 이를 **멀티 헤드 어텐션(Multi-Head Attention)** 이라고 한다.  

### 층 정규화(Layer Normalization)
딥러닝에서 여러개 층을 거치며 학습할 때, 학습의 속도를 높이기 위해 고안된 **배치 정규화(Batch Normalization)** 이 있다.  
배치 정규화 방식은 모든 샘플에서 특정 채널의 데이터를 모아 평균과 분산을 계산한 후, 정규화를 적용한다. 이는 학습 속도를 향상시키고, 학습 구조를 안정화 시키는 효과가 있다.  
그러나, 텍스트 데이터는 샘플마다 길이가 다르기 때문에 배치 정규화를 적용하기 어려웠고, 이를 위해 **층 정규화(Layer Normalization)** 이 고안되었다,  
<br />  

> <img width="394" height="98" alt="image" src="https://github.com/user-attachments/assets/40c4106e-ae4b-4d6b-9162-72f50f3a8e96" />   
층 정규화는 각 샘플의 토큰마다 개별적으로 정규화를 수행한다.  
문장의 개별 토큰의 모든 특성(임베딩 차원 = 뉴런 개수 = H)의 값의 평균과 표준편차를 구한다.
<br />  

### 잔차 연결(Residual Connection)
> <img width="859" height="335" alt="image" src="https://github.com/user-attachments/assets/1a97a224-30b4-4d12-a170-e59e5fce3005" />   
각 서브 레이어의 입력값을 서브 레이어의 출력값에 그대로 더해주는 연결이다.
네트워크가 깊어져도 정보가 잘 전달되기 하고, 학습을 용이하게 한다.
트랜스포머에서는 멀티헤드 어텐션과 층 정규화 사이에 추가되어 쓰인다.

### 피드포워드 네트워크(Feedforward Network)
트랜스포머는 멀티헤드 어텐션과 층 정규화 다음에 밀집층을 이용하여 비선형 변환을 수행한다.  
이러한 밀집층을 종종 피드포워드 네트워크라고 부른다.  
피드포워드 네트워크는 보통 두 개의 밀집층으로 구성된다. 첫 번쨰 밀집층은 ReLU 활성화 함수를 사용하며,  
두 번째 밀집층은 활성화 함수를 사용하지 않는다. 이후 드롭아웃 층이 추가되며, 이 세 개의 층으로 구성된 서브 레이어를 또 다른 잔차 연결이 감싼다.  

### 토큰 임베딩과 위치 임베딩
트랜스포머 역시 토큰을 고정된 크기의 실수 벡터로 변환하여 임베딩을 수행한다.  
그러나 기존 모델과 달리 순차적으로 토큰을 처리하지 않고 모든 토큰을 한 번에 처리하기에, 토큰의 위치 정보를 반영하지 않는 문제가 생긴다.  
<br />

**위치 임베딩(Positional Embedding)**  
위와 같은 문제를 해결하기 위해 고안된 방법으로,  
sin 함수와 cos 함수를 사용해 토큰의 위치에 따라 변하는 벡터를 생성하고 이를 단어 임베딩에 더하는 방식이다.  
계산 과정은 다음과 같다.  
> <img width="702" height="478" alt="image" src="https://github.com/user-attachments/assets/69a77105-4c86-4774-969b-8a58ac10258b" />

### 인코더 블록
위와 같은 구성 요소들을 종합한 트랜스포머의 인코더 블록은 다음과 같다.
> <img width="219" height="279" alt="image" src="https://github.com/user-attachments/assets/ae13b6a9-7758-4741-a260-8a69121a139b" />
트랜스포머에서는 이러한 인코더 블록이 여러 개 연결되어 사용된다.  

### 디코더 블록
> <img width="250" height="385" alt="image" src="https://github.com/user-attachments/assets/68747676-f74f-4316-bbc2-1bb4c6dd2451" />  

디코더 블록의 기본 구조 또한 인코더 블록과 유사하나, 몇 가지 차이점이 존재한다.  
바로  인코더가 출력한 임베딩 벡터를 입력으로 받는 멀티 헤드 어텐션 층이 존재한다는 것이다.  
이 층은 디코더에서 받은 벡터를 Query로 사용하고, 인코더의 출력을 Key와 Value로 사용한다.  
이러한 방식을 **크로스 어텐션(Cross Attention)** 이라고 한다.  

디코더 블록 역시 인코더 블록과 마찬가지로 여러 개가 반복적으로 쌓여 전체 디코더 모델을 구상한다.  
따라서 인코더 블록의 출력은 첫 번째 디코더 블록뿐만 아니라 모든 디코더 블록에 전달된다.  
이때, 마지막 인코더 블록의 출력만을 사용한다.  
<br />  

디코더 역시 훈련 과정에서 모델의 출력과 정답을 비교하여 손실을 줄이는 방향으로 학습을 한다.  
그러나 디코더가 다음에 출력할 정답을 미리 알게될 경우 올바른 학습이 이루어질 수 없다.  
이러한 문제를 방지하기 위해 디코더의 첫 번째 멀티 헤드 어텐션 층에서는 원본 토큰 입력에 마스킹(Masking) 처리를 한다.  
즉, 디코더가 한 타임스텝에서 어텐션 점수를 계산할 때 현재 토큰까지만 참고하고, 이후의 토큰은 볼 수 없도록 제한한다.  
이 때문에 디코더 블록의 첫 멀티 헤드 어텐션 층을 **마스크드 멀티 헤드 어텐션(Masked Multi-Head Attention)이라고 부른다.  

### 트랜스포머의 전체 구조
트랜스포머의 전체적인 구조도는 다음과 같다.  
> <img width="500" height="614" alt="image" src="https://github.com/user-attachments/assets/dc67ea35-bb66-4a49-b53e-637f3de11e64" />


