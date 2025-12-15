## 핵심 키워드
> **결정 트리(decision tree)**: 데이터의 특성을 질문(기준)으로 하여 좌, 우로 샘플들을 분류하는 알고리즘이다.
> 결정 트리는 여러개의 노드로 구성되어 있다. 노드는 훈련 데이터의 특성에 대한 테스트를 표현한다. 가장 위의 노드를 루트 노드, 맨 아래 끝의 노드를 리프 노드라 한다.
> 결정 트리는 이해하기 쉽고 설명에 용이한 구조를 가지고 있다.

> **불순도(impurity)**: 불순도는 트리의 각 노드가 가지고 있는 값으로, 여러 클래스가 섞여 있을수록 그 수치가 높다.
> 일반적으로 쓰이는 불순도는 지니(gini) 불순도와 엔트로피(entrophy) 불순도가 있다. 각각의 계산 식은 다음과 같다.
> <img width="388" height="112" alt="image" src="https://github.com/user-attachments/assets/06ca0143-c73b-4f49-b560-5db44d60f596" />
> <img width="197" height="79" alt="image" src="https://github.com/user-attachments/assets/6bb8024b-6722-48c7-a7e9-d257eca6f682" />
