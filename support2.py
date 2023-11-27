import networkx as nx
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.io import imread

#weka segmentation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import skimage
from skimage.io import imread
import warnings
warnings.simplefilter('ignore', np.RankWarning)
import joblib
import statistics
import cv2
from skimage.morphology import skeletonize
from skimage import data
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage import img_as_ubyte
import math
import networkx as nx
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from scipy.optimize import leastsq
from sklearn.cluster import DBSCAN
from scipy.optimize import leastsq
from scipy.ndimage import distance_transform_edt, sobel

def remove_if_not_touching(ind,dist,diameters):
    #print(dist)
    new_ind = {}
    new_dist = {}
    for i, item in enumerate(ind):
        new_ind[i] = []
        new_dist[i] = []
        for j, distance in enumerate(dist[i][0:]):
            combinedDiameter = 1.2*((diameters[i]/2)+(diameters[item[j]]/2))
            #print(combinedDiameter," ", distance," ",i, "to", ind[i][j])
            if (distance <= combinedDiameter) and (not j == 0):

                new_ind[i].append(str(item[j]))
                new_dist[i].append(str(distance))
    return new_ind,new_dist


#sees if nodes are truly adjacent, then links them in a graph
def nodeComparator(node1, node2, graph):
    if node1 == node2:
        return

    Positions = nx.get_node_attributes(graph,'pos')

    Diameters = nx.get_node_attributes(graph,'d')
    DeltaXY = (Positions[node2][0]-Positions[node1][0])**2 + (Positions[node2][1]-Positions[node1][1])**2
    Comparator = 1.25*(Diameters[node2]/2+Diameters[node1]/2)**2-DeltaXY

    if Comparator >= 0:
        graph.add_edge(node1, node2)
        #print("edge added between nodes: ", node1, ", ", node2)

    return

def mergeDictionaries(dictionary1, dictionary2):

    return dictionary1


class graphAnalysis:
    def __init__(self, node_list, img):
        self.G = nx.Graph()
        self.nodelist = node_list
        self.img = img
        self.fileLocation = img

    def startup(self,imageSave,massFract):
        img = imread(self.img)#[0][:][:]
        #print(img.shape)
        #plt.imshow(img[0][:][:])

        self.G.add_nodes_from(self.nodelist)
        pos1 = nx.get_node_attributes(self.G, "pos")
        diameters = nx.get_node_attributes(self.G, "d")
        #print(diameters)
        maxDiamter = max(list(diameters.values()))
        ImageDimension = img.shape
        imageXDim = ImageDimension[1]
        imageYDim = ImageDimension[0]

        LL = list(pos1.values())
        #print(LL)

        #iterate through all nodes as sets
        allnodes = set(self.G.nodes)
        allnodesNotSet = list(self.G.nodes)



        #t1 = time.time()



        tree = BallTree(LL, leaf_size=2, metric = 'euclidean')
        ind,dist = tree.query_radius(LL, r=maxDiamter*1.5, return_distance=True, sort_results=True)
        #print(list(ind[14]))
        #print(list(dist[14]))
        ind,dist = remove_if_not_touching(ind,dist,diameters)


        indexer = []
        neighbors = []
        distances = []
        counts = []


        for index1,node1 in enumerate(allnodesNotSet):

            for j in ind[index1]:
                self.G.add_edge(node1, allnodesNotSet[int(j)])




       # t2 = time.time()

        #print(str(t1-t0)," ", str(t2-t1))
        position =  nx.get_node_attributes(self.G,'pos')
        #print(position)



        #print(allnodes)
        #originally will contain all nodes, but will reduce in size as nodes are put in appropriate subtree
        nodescovered = set()
        #collection of nodes for each subgraph
        subgraphs = []

        #finding degrees per node
        Nodedegrees = [node for node in self.G.degree()]
        Nodedegrees = dict(Nodedegrees)


        localdegreedictionary = {}
        colordictionary = {}

        #color code stuff
        bluecol = 255
        greecol = 255
        redcol = 0

        #will do two things, make subgraphs and also find local degrees
        for nodei in allnodes:
            #local degrees
            nodesadlisti = list(nx.neighbors(self.G,nodei))
            nodeDegreeNumber = Nodedegrees[nodei]

            if nodesadlisti:
                n = 0
                sumi = 0
                for adjacentnode in nodesadlisti:
                    n += 1
                    sumi += Nodedegrees[adjacentnode]
                nodeDegreeNumber = (nodeDegreeNumber + sumi/n)/2


            colordictionary[nodei] = "#00" + format(int(255-min(max((nodeDegreeNumber-2)/4*255,0),255)), '02x') +format(int(min(max((nodeDegreeNumber-2)/4*255,0),255)), '02x')
            if nodeDegreeNumber == 0:
                colordictionary[nodei] = "#FF0000"
            localdegreedictionary[nodei] = nodeDegreeNumber




            #subgraph shit
            if nodei in nodescovered:
                continue
            setofnodesini = nx.node_connected_component(self.G, nodei)
            nodescovered = nodescovered.union(setofnodesini)
            #print(nodescovered)
            subgraphs.append(setofnodesini)





        coloriterable = 0
        #colordictionary = {}
        plt.figure(figsize=(20, 20))

        #print(localdegreedictionary)





        #for setx in subgraphs:

        #    colordictionary.update(dict.fromkeys(setx, colors[coloriterable]))

        #    coloriterable = (coloriterable+1)%3

            #print(coloriterable)
        colorlist = [colordictionary[nodej] for nodej in allnodes]
        degreesMean = [localdegreedictionary[nodej] for nodej in allnodes]



        #need to combine number of neighbors dict, x,y positions and node number
        newDict = {}
        for element in Nodedegrees:
            newDict[element] = [position[element][0],position[element][1],Nodedegrees[element]]

        NodeDf = pd.DataFrame.from_dict(newDict, columns = ["x","y","degree"], orient = "index")


        Output1 =  StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(NodeDf)



        ScaledNodeDF = pd.DataFrame(Output1, columns = ["x","y","degree"])
        ScaledNodeDF_degree = ScaledNodeDF.copy()
        #print(ScaledNodeDF_degree)
        ScaledNodeDF_degree= ScaledNodeDF_degree.drop(["x","y"], axis = 1)

        #PCA
        #pca = PCA(n_components = 2)
        #pCompTables = {}
        #pca.fit(ScaledNodeDF)
        #pComps = pca.transform(ScaledNodeDF)
        #pCompTable = pd.DataFrame(pComps[:, [0,1]], columns = ['pc1', 'pc2'])
        #print(pca.explained_variance_ratio_.sum())
        #print(pCompTable)

        #a2 = sns.relplot(
        #    data=pCompTable, kind = "scatter", aspect = 1.5,
        #    x="pc1", y="pc2",   palette = "flare",facet_kws=dict(despine=False), s = 100, edgecolor = "black"
        #    );


        #plt.show()

        #k means



        #print(colorlist)
        #area = 0
        #for diameter in diameters:
        #    area += (diameter**2)*3.14/4
        Void = 0
        #print(list(Nodedegrees.values()))
        #print(degreesMean)
        NodedegreesMean = np.mean(list(Nodedegrees.values()))
        #print(NodedegreesMean)
        outputValues = [NodedegreesMean,Void]



        #print(colorlist)

        #print(colordictionary)


        plt.imshow(img, cmap = 'gray')
        #nx.draw(self.G)

        nx.draw_networkx(self.G, pos=position,  node_color=colorlist, edge_color = "blue", width = 4)
        #nx.draw(G, pos=position, with_labels = True, node_color="blue", edge_color = "blue", width = 4)

        plt.gca().invert_yaxis()

        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        # Create a PIL image object from the BytesIO object
        img_with_graph = Image.open(buf)
        img_with_graph.save(imageSave,"PNG",bbox_inches='tight', pad_inches=0)
        del img_with_graph
        buf.close()
        del buf
        #buf.flush()
        #for i in [2,3]:
            #i = 2
            #kmeans =  KMeans(n_clusters = i)
            #print("k is equal to ", i)
            #print(ScaledNodeDF_degree)
            #kmeans.fit(ScaledNodeDF_degree)

            #ScaledNodeDF["Lables"] = kmeans.labels_

            #plt.show()


        return [0,NodedegreesMean,Void,massFract]

    def nodeAnalysis(self):
        pass
