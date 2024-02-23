import numpy as np

class KMedoids:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.medoids = None

    def fit(self, X):
        # Inicialización aleatoria de medoides
        self.medoids = X[np.random.choice(range(len(X)), self.n_clusters, replace=False)]
        dif_medoid = float('inf')  # Inicializar con un valor grande

        while dif_medoid > 1*10**(-5):
            # Asignación de puntos al medoide más cercano
            distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Actualización de medoides
            new_medoids = np.array([X[labels == j][np.argmin(np.sum(distances[labels == j], axis=1))] for j in range(self.n_clusters)])

            # Calcular la diferencia entre los centroides antiguos y nuevos
            dif_medoid = np.linalg.norm(self.medoids - new_medoids)
            
            self.medoids = new_medoids
        return self.medoids
    
    def predict(self, X):
        if self.medoids is None:
            raise ValueError("El modelo no ha sido ajustado. Por favor, llama al método fit primero.")

        # Calcular distancias entre puntos y medoides
        distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2)

        # Asignar cada punto al clúster del medoide más cercano
        labels = np.argmin(distances, axis=1)

        return labels