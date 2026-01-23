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
* Query(Q): "무엇을 찾고 있는가" (비교 기준)
* Key(K): "각 토큰이 어떤 정보를 갖고 있는가"  (비교 대상의 설명)
* Value(V): "실제로 전달할 정보"  (전달 정보)

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

