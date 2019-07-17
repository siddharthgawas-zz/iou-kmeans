import pandas as pd
import numpy as np
from mini_batch_kmeans import MiniBatchKMeans
from util import np_cvt_coord_to_mid_point, np_cvt_coord_to_diagonal
import click

@click.command()
@click.argument('filename',type=click.Path(exists=True))
@click.argument('k',type=int)
@click.option('-n','--max-iteration','n',default=1000,show_default=True,type=int,help='Max iteration')
@click.option('-b','--sample-size','b',default=-1,show_default=True,type=int,help='Mini batch size for K-means')
@click.option('-s','--scale','s',default=1,show_default=True,type=int,help='Scaling factor')
@click.option('-o','--output','output_file',default='anchor.txt',show_default=True,type=str,help='Output file name')
def main(filename,k,n,b,s,output_file):
    """
    Utility generates K anchor boxes using input CSV file FILENAME.
    """
    DATA_PATH = filename
    OUT_FILE = output_file
    K = k
    MAX_ITER = n
    SAMPLE_SIZE = b

    scale = s
    train_data = pd.read_csv(DATA_PATH)
    if SAMPLE_SIZE == -1:
        SAMPLE_SIZE = train_data.shape[0]
    df = train_data[['w','h']]
    data = np.array(df)

    k_means = MiniBatchKMeans(K,max_iteration=MAX_ITER,mini_batch_size=SAMPLE_SIZE)
    avg_error,avg_iou = k_means.train(data,iteration_hist=True)
    cluster_vectors = k_means.cluster_vectors
    cluster_vectors = cluster_vectors.reshape((-1,))
    cluster_vectors = scale * cluster_vectors
    print('Cluster Vectors = ', cluster_vectors)
    with open(OUT_FILE,'w') as f:
        f.write(str(cluster_vectors.tolist()))
        
if __name__ == '__main__':
    main()