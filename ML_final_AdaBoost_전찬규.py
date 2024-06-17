import math
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class AdaBoost:
    def __init__(self, base_learner, T):
        self.base_learner = base_learner
        self.T = T
        self.alphas = []
        self.learners = []

    def fit(self, X, y):
        m = len(y)
        D = [1/m]*m # 샘플 가중치 분포 초기화
        
        for t in range(self.T):
            h_t = self.base_learner(X, y, D) # base learner training
            
            # h_t의 오차 계산!!!!!!!!!!!!
            error = sum(D[i] for i in range(m) if h_t(X[i]) != y[i])
            
            # 오차가 너무 클 경우 h_t 버림
            if error > 0.5:
                break
            
            # 분류기 h_t의 가중치 결정
            alpha_t = 0.5 * math.log((1 - error) / error)
            self.alphas.append(alpha_t)
            self.learners.append(h_t)
            
            # 샘플분포 갱신
            Z_t = sum(D[i] * math.exp(-alpha_t if h_t(X[i]) == y[i] else alpha_t) for i in range(m))
            D = [(D[i] / Z_t) * math.exp(-alpha_t if h_t(X[i]) == y[i] else alpha_t) for i in range(m)]
        
    def predict(self, X):
        # Output
        final_predictions = [0] * len(X)
        for alpha, learner in zip(self.alphas, self.learners):
            for i in range(len(X)):
                final_predictions[i] += alpha * learner(X[i])
        
        return [1 if pred > 0 else -1 for pred in final_predictions]
    
def Base_learner(X, y, D):
    best_base = None
    min_error = float('inf')
    
    for threshold in [x[0] for x in X]:
        for sign in [-1, 1]:
            base = lambda x: sign * (1 if x[0] >= threshold else -1)
            error = sum(D[i] for i in range(len(y)) if base(X[i]) != y[i])
            if error < min_error:
                min_error = error
                best_base = base
    
    return best_base

## Example
iris = load_iris()
X = iris.data
y = iris.target

# 이진분류를 위해 2개 클래스만 선택
X = X[y != 2]
y = y[y != 2]

# 클래스를 -1과 1로 변경
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델링
ada = AdaBoost(Base_learner, 10)
ada.fit(X_train, y_train)

# Predict
predictions = ada.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("Predictions:", predictions)
print("True labels:", y_test)
