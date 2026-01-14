# LSTM(Long Short-Term Memory)
## LSTM의 구조
> <img width="552" height="285" alt="image" src="https://github.com/user-attachments/assets/e959dc77-91c8-4002-9421-ce970790291a" />


LSTM은 은닉 상태(Hidden State, 이하 ht)외에 셀 상태(Cell State, 이하 ct)를 추가로 가진다.  
* **셀 상태(Cell State)**   
  * 타임스텝을 따라 거의 선형적으로 흐르며 정보 보존   
  * 필요 없는 정보는 제거하고, 중요한 정보만 유지하도록 게이트가 제어  

* **은닉 상태(Hidden State)**
  * 외부로 출력되는 상태   
  * 현재 시점의 요약 정보   
  * 다음 LSTM 셀 및 출력층으로 전달됨  
  
LSTM의 핵심은 cell state ct를 선형 경로로 유지하면서, gate를 통해 정보의 흐름을 조절하는 데 있다.  

LSTM 셀은 특정 기능을 가진 뉴런 집합들로 구성된다. 뉴런 집합은 역할에 따라 4종류로 나뉜다.  
1. Forget Gate: 이전 기억을 얼마나 유지할지 조절함  
2. Input Gate: 새 정보를 반영할 비율을 조절함  
3. Output Gate: 외부로 노출할 정보를 조절함
4. Candidate: 현재 시점에 새로 추가될 수 있는 기억의 내용을 생성함  

* Cadndiate가 새로운 내용을 제안    
  -> 입력 xt와  ht-1를 바탕으로 현재 시점에서 유용하다고 판단되는 정보 벡터 생성    
* Input Gate와 곱해져 선택적으로 Cell State에 반영    
* Forget Gate가 Cell State의 각 성분에 대해 유지 비율(0~1)을 결정하는 제어 신호를 생성  
  -> 현재 입력 정보에 대해 이전 기억중 어떤 부분은 이제 의미가 없는지를 학습  

gate, 혹은 candidate라는 이름으로 부르고 있으나, 이들은 모두 뉴런의 집합이다.  
이 객체들을 구성하는 개별 요소는 weight와 bias로 입력에 대한 선형 변환을 수행하고 활성화 함수를 적용하는 뉴런이다.  

## LSTM의 기능 단위  
> LSTM의 기능 단위는 3개의 gate와 candidate이며, hidden 차원마다 하나씩 존재한다.  
> 즉, unit = 128이면, 각 기능 단위들 또한 128개씩 존재한다.  
> LSTM의 units는 모델이 시퀀스를 표현하기 위해 사용하는 잠재 상태 공간의 차원으로,  
> 정보 해석 관점에서는 동시에 추적할 수 있는 잠재 특성의 수로 비유할 수 있다.  
> hidden state 차원마다 시퀀스를 표현하는 축이 다르기 때문에, 모든 gate와 candidate는 서로 다른 가중치와 절편을 가진다.  
> LSTM이 주어진 시계열 데이터를 다차원적으로 해석하고 관리할 수 있게 만드는 원리인 것이다.

> <img width="322" height="235" alt="image" src="https://github.com/user-attachments/assets/dbeb1af9-6c62-4a60-8f20-642d1299642f" />



### 1.망각 게이트(Forget Gate)
> <img width="537" height="335" alt="image" src="https://github.com/user-attachments/assets/902da107-9049-4b82-afd2-1e7a66159734" />   

망각 게이트는 기존에 가지고 있던 정보, 즉 장기 기억을 얼마나 유지할지 조절(삭제)하는 역할이다.  

### 2.입력 게이트(Input Gate)
> <img width="524" height="180" alt="image" src="https://github.com/user-attachments/assets/8928adfb-955c-4baa-887a-895abf39f9b0" />

### 3.후보 기억(Candidate Cell State)
> <img width="539" height="287" alt="image" src="https://github.com/user-attachments/assets/05682dd5-935f-448e-8556-a71afd857035" />

### 4.Cell State 업데이트 
> <img width="506" height="387" alt="image" src="https://github.com/user-attachments/assets/017b72aa-46fc-49ce-8e7a-098415899728" />

후보 기억이 생성되고 실제로 셀 상태에 반영될 때, 입력 게이트의 제어를 받는 것을 알 수 있다.  

### 5.출력 게이트(Output Gate)
> <img width="527" height="152" alt="image" src="https://github.com/user-attachments/assets/6c8ecef6-2698-4f43-8d5a-4bc74ede461a" />

### 6.Hidden State 계산
> <img width="489" height="204" alt="image" src="https://github.com/user-attachments/assets/b9ef5de2-eecc-41bf-a2da-a0ef2320a69b" />

### 7.전체 흐름 요약(한 타임스텝)
> <img width="260" height="168" alt="image" src="https://github.com/user-attachments/assets/4787cac2-16d5-4809-b99f-d496065cd47e" />  

### 각 gate들이 기능을 갖게 되는 이유(핵심 관점)
> <img width="385" height="201" alt="image" src="https://github.com/user-attachments/assets/9d5edcd6-58f5-4255-ba2b-572a6939e05c" />  

각 gate들이 특정 기능을 수행할 수 있는 이유는 하드 코딩이 아니라,  
**손실함수를 최소화하는 방향으로 학습된 가중치 구조** 덕분이다.


# GRU(Gated Recurrent Unit)
## GRU의 구조
> <img width="862" height="608" alt="image" src="https://github.com/user-attachments/assets/01c7a34b-f573-4e0b-9c2c-e25f42bb2b05" />

GRU(Gated Recurrent Unit)는 LSTM의 복잡도를 줄이면서도 장기 의존성(long-term dependency)을 효과적으로 학습하기 위해 제안된 순환 신경망 구조이다.  
핵심적인 특징은 cell state를 별도로 두지 않고, 두 개의 게이트만으로 은닉 상태(hidden state)의 흐름을 제어한다는 점이다.  
LSTM보다 가중치가 적기 때문에 계산량이 적지만, LSTM 못지 않은 성능을 낸다.  

GRU의 핵심 구성 요소는 다음과 같다.
1. Update Gate(갱신 게이트)
2. Reset Gate(리셋 게이트)
3. Candidate Hidden State(후보 은닉 상태)

## GRU의 기능 단위
### 1.갱신 게이트(Update Gate)
> <img width="527" height="161" alt="image" src="https://github.com/user-attachments/assets/bcea2bf1-c75f-4063-8c93-d2e62d33de19" />

LSTM의 Forget Gate와 Input Gate의 기능이 합쳐졌다고 보면 된다.    
정보 흐름의 주 스위치이며, 장기 의존성 보존의 핵심 요소이다.  
### 2.리셋 게이트(Reset Gate)
> <img width="523" height="140" alt="image" src="https://github.com/user-attachments/assets/5afa49fd-bd4e-4dc5-8498-462bdee11de6" />

과거를 초기화할지 여부를 결정한다.  
문장 경계와 상태 전환 등, 국소적 패턴 학습에 유리하다.

### 3.후보 은닉 상태(Candidate Hidden State)
> <img width="569" height="111" alt="image" src="https://github.com/user-attachments/assets/27e4221a-6bfc-4265-aa97-e8e48f884d6f" />

### 4.최종 Hidden State
> <img width="539" height="113" alt="image" src="https://github.com/user-attachments/assets/6fa2e33d-e6c7-49f7-a698-4cc157820a0c" />

GRU에서는 hidden state 하나가 곧 메모리이며, 출력과 내부 상태가 분리되지 않는다.  

# 게이트의 역할 분화 원리
LSTM과 GRU를 비롯한 CNN 또한 각 뉴런들의 기본 원리는 다른 신경망들과 같다.  
**출력을 바탕으로 손실을 계산하고, 이를 역전파하여 가중치를 갱신한다.**  
<br />   

게이트의 역할이란,   
**특정 입력 패턴에서 특정 게이트가 일관되게 열리거나 닫히는 방향으로 파라미터가 수렴하는 현상을 의미한다.**  
이 역할은 **학습 과정 전체의 동역학**에 의해 결정된다.  
<br />  

LSTM과 GRU의 모든 뉴런들은 가중치 벡터가 서로 다르며, 계산 과정에서 수식적으로 개입하는 위치가 다르다.  
이로 인해 같은 입력, 역전파를 받더라도 각 파라미터의 gradient 벡터는 다르게 나타난다.  
> <img width="545" height="321" alt="image" src="https://github.com/user-attachments/assets/0f97939d-f123-4c11-a5b9-b6d08dfb6696" />  
<br />  

이 계산 그래프 내 위치 차이가 게이트가 특정 역할을 하도록 강제하는 **구조적 편향(Structural Bias)** 을 만든다.  
이러한 구조에서 각 파라미터가 학습 과정에서 서로 다른 보상을 받아 경향성이 강화되어 고착되면, 역할이 분화된다.  
<br />

초기화 단계에서는 다음과 같은 조건을 지키는 내에서 무작위로 가중치를 초기화 한다.    
1. 대칭성 붕괴  
2. 게이트 간 파라미터의 상이성  
3. 포화(saturation) 회피  
4. 초기 sigmoid의 출력이 0이나 1것을 방지  

이는 게이트 구조를 만드는 것이 아니라, 학습 과정에서 역할의 분화가 일어나지 않는 것을 막기 위함이다.



