# Part IV Project: 3D Human Motion Capture
## Setup
**Step 1:** Git clone repo
```
git clone https://github.com/akwa749/P4P
```

**Step 2: (Required)** Creates conda environment with required dependencies and versioning
```
conda env create -f env.yml
```

Additional commands to help:
```
# list all the conda environment available
conda info --envs   

# list all the dependencies and versioning in current conda environment
conda list

# Create new environment named as `envname`
conda create --name envname

# Remove environment and its dependencies
conda remove --name envname --all

# Clone an existing environment
conda create --name clone_envname --clone envname
```

<!-- ## Errors: SSL Error 
copy the following files from CONDA_PATH\Library\bin to CONDA_PATH\DLLs
```
libcrypto-1_1-x64.*
libssl-1_1-x64.*
```
--- -->

---
centroid cluster.py
is the file that is currently working

## notes 6/05/2023  

- need to try use DB scan as better for separating clusters of different shapes and sizes?
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(point_cloud)

changing eps and min_samples to find optimal values for the sample

- can also try pre processing the clustering algorithm by removing points that are far away from the centroid
to reduce impacts of te backhground on the clustering results

def filter_points_far_from_centroid(point_cloud, centroid, threshold):
    distances = np.linalg.norm(point_cloud - centroid, axis=1)
    return point_cloud[distances < threshold]

filtered_point_cloud = filter_points_far_from_centroid(point_cloud, centroid, threshold)
labels = kmeans.fit_predict(filtered_point_cloud)



--update from 11/05/2023
Becareful of using DBscan computatively expensive will crash computer, tried using lower epislon value and higher
sample size as that makes it compute faster but doesn't work maybe need to debug?


--update from 15/05/2023
scrap DBSCAN fucks computer over useless.
Goal should be remove far away points which the boundingbox method.py does and then most importantly remove the floor from that potentially using clustering or techniques in that same file i was exploring. And ideally also find a way to mathematically caculate the padding we use for it to be fully automatic but that can be next week problem or something