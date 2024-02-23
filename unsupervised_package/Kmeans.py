import numpy as np

class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroids = None

    def fit(self, X):
        # Inicialización aleatoria de centroides
        self.centroids = X[np.random.choice(range(len(X)), self.n_clusters, replace=False)]
        dif_centroide = float('inf')  # Inicializar con un valor grande

        while dif_centroide > 1*10**(-5):
            
            # Asignación de puntos al centroide más cercano
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Actualización de centroides
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])

            # Calcular la diferencia entre los centroides antiguos y nuevos
            dif_centroide = np.linalg.norm(self.centroids - new_centroids)

            self.centroids = new_centroids
            return self.centroids
        
    def predict(self, X):
        if self.centroids is None:
            raise ValueError("El modelo no ha sido ajustado. Por favor, llama al método fit primero.")

        # Calcular distancias entre puntos y centroides
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

        # Asignar cada punto al cluster del centroide más cercano
        labels = np.argmin(distances, axis=1)

        return labels
