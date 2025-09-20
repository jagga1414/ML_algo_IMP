import numpy as np



class kmeans():
    def __init__(self,k=3):
        self.clusters = []
        self.k = k

    def initialize(self,x):
        for i in range(self.k):

            idx = np.random.randint(0, x.shape[0])
            point = x[idx]

            cluster = {"points":[],
                    "centroid": point
                    }
            self.clusters.append(cluster)

    def distance(self,p1,p2):
        return np.sqrt(np.sum((p1-p2)**2))

    def assignClusters(self,x):

        for data_point in x:

            dist = []

            for cluster in self.clusters:

                dist.append(self.distance(data_point,cluster["centroid"]))
            

            cluster_num = np.argmin(dist)
            self.clusters[cluster_num]["points"].append(data_point)
    
    def assignClusterToPoint(self,data_point):
        dist = []
        for cluster in self.clusters:

            dist.append(self.distance(data_point,cluster["centroid"]))

        cluster_num = np.argmin(dist)
        return cluster_num

    def updateCentroid(self):
        
        for cluster in self.clusters:

            points = cluster["points"]

            mean = np.mean(points,axis=0)
            cluster["centroid"] = mean
            cluster["points"]  = []

    def fit(self,max_itr,x):
        self.initialize(x)
        for i in range(max_itr):
            self.updateCentroid()
            self.assignClusters(x)
        
    def predict(self,x):

        return self.assignClusterToPoint(x)

    

        


