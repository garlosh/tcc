import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

def MCLS(X_labeled, y_labeled, X_unlabeled, max_iter=50):
    """
    Implementação do MCLS para rótulos {0, 1}, recebendo:
      - X_labeled:    matriz (numpy ou scipy.sparse) com as features dos dados rotulados
      - y_labeled:    pandas.Series com rótulos em {0, 1}
      - X_unlabeled:  matriz (numpy ou scipy.sparse) com as features dos dados não rotulados
      - max_iter:     número máximo de iterações

    Retorna:
      - Uma função 'classificador' (callable) que recebe X e devolve array de rótulos {0,1}.
    """
    # Passo 1: Inicializar pseudo-rótulos (aqui, todos como 0).
    n_unlabeled = X_unlabeled.shape[0]
    unlabeled_y = np.zeros(n_unlabeled, dtype=int)

    for _ in range(max_iter):
        # Concatenar X_labeled e X_unlabeled
        X_train = _stack_features(X_labeled, X_unlabeled)
        # Concatenar rótulos reais + pseudo-rótulos
        y_train = np.concatenate([y_labeled.values, unlabeled_y])
        
        # Treinar LS-SVM (SVC linear) nos dados rotulados + pseudo-rotulados
        w, b = train_ls_svm(X_train, y_train)
        
        # Obter novos pseudo-rótulos
        new_unlabeled_y = threshold_predict_all(w, b, X_unlabeled)
        
        # Checar convergência
        if np.array_equal(new_unlabeled_y, unlabeled_y):
            break
        unlabeled_y = new_unlabeled_y

    # Retorna uma função que, dada uma matriz X, prevê {0,1}
    return lambda X: threshold_predict_all(w, b, X)


def train_ls_svm(X, y):
    """
    Treina um SVC linear (similar ao LS-SVM) e retorna (w, b):
      - w: vetor de pesos (1D)
      - b: bias (escalar)
    """
    model = SVC(kernel='linear', C=1.0)
    model.fit(X, y)
    w = model.coef_[0]      # shape (n_features,)
    b = model.intercept_[0] # escalar
    return w, b


def threshold_predict_all(w, b, X):
    """
    Aplica (w·X + b) e usa limiar 0.0 para mapear em {0,1}.
    """
    if sparse.issparse(X):
        scores = X.dot(w) + b
    else:
        scores = np.dot(X, w) + b

    return (scores >= 0).astype(int)


def _stack_features(X1, X2):
    """
    Empilha (verticalmente) duas matrizes X1 e X2, que podem ser
    numpy arrays ou scipy.sparse, assegurando shape compatível.
    """
    if sparse.issparse(X1) and sparse.issparse(X2):
        return sparse.vstack([X1, X2])
    else:
        return np.vstack([_to_dense_if_needed(X1), _to_dense_if_needed(X2)])


def _to_dense_if_needed(X):
    """
    Converte matriz esparsa para array denso, se necessário.
    """
    if sparse.issparse(X):
        return X.toarray()
    return X


# =====================================
# ============ DEMONSTRAÇÃO ===========
# =====================================
if __name__ == "__main__":
    # 1) Gerar dados sintéticos com make_classification
    #    Vamos gerar 100 amostras (X, y). E depois
    #    separar 70 como "rotuladas" e 30 como "não rotuladas".
    X, y = make_classification(
        n_samples=100, n_features=5,  # 5 features
        n_informative=3, n_redundant=0,
        n_classes=2, random_state=42
    )

    # Em um cenário real, você não teria y para as não rotuladas.
    # Aqui vamos "esconder" apenas para checar no final.
    
    # 2) Separar parte rotulada (70) e parte não rotulada (30)
    X_labeled = X[:70, :]    # shape (70, 5)
    y_labeled = y[:70]       # shape (70,)
    X_unlabeled = X[70:, :]  # shape (30, 5)
    y_unlabeled_true = y[70:]# shape (30,) - APENAS para avaliarmos no final

    # 3) Transformar y_labeled em Series do pandas
    y_labeled_series = pd.Series(y_labeled)

    # 4) Aplicar MCLS
    classifier = MCLS(X_labeled, y_labeled_series, X_unlabeled, max_iter=20)

    # 5) Obter pseudo-rótulos para as 30 amostras não rotuladas
    y_unlabeled_pred = classifier(X_unlabeled)

    # 6) (Opcional) avaliar a qualidade desses pseudo-rótulos (como sabemos a verdade)
    acc_unlabeled = accuracy_score(y_unlabeled_true, y_unlabeled_pred)
    
    print("Pseudo-rótulos preditos para as amostras não rotuladas:")
    print(y_unlabeled_pred)
    print(f"\nAcurácia nos (30) dados 'não rotulados' (para demonstração): {acc_unlabeled:.2f}")

    # 7) Usar o classificador para prever em novos dados
    #    (Exemplo: vamos prever nas mesmas 5 primeiras amostras do dataset)
    X_test = X[:5, :]
    y_test_pred = classifier(X_test)
    print("\nPredições em 5 novos dados (primeiras amostras do dataset):", y_test_pred)
