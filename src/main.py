import support
from support import *
from support2 import *
import glob,os
from PIL import Image, ImageDraw
listofString = glob.glob("I:/My Drive/Research/OtherProjects/OilInwaterStuff/ConfocalRheo2/ConfocalRheo-newprotocal/Output/CDA1ALL/*.tif")
#location = "I:/My Drive/Research/OtherProjects/OilInwaterStuff/ConfocalRheo2/ConfocalRheo-newprotocal/Output/CDA1ALL/"
#outloc = "I:/My Drive/Research/OtherProjects/OilInwaterStuff/ConfocalRheo2/ConfocalRheo-newprotocal/Output/CDA1OUT1/"
#outloc2 = "I:/My Drive/Research/OtherProjects/OilInwaterStuff/ConfocalRheo2/ConfocalRheo-newprotocal/Output/CDA1OUT2/"

location = "../SplitData/500-0.5/"
outloc2 = "../SplitData/500-0.5-out/"






import pandas as pd
import gc


class imageParser():
    def __init__(self, filelocation,outlocation):
        #print(filelocation)
        #self.Listfiles =  glob.glob(filelocation+"*.tif",recursive=False)
        self.fileDict = {}
        self.outloc = outlocation
        self.sampleName = filelocation.split("/")[-2]
        for entry in os.scandir(filelocation):
            if entry.is_file():
                if ".tif" in entry.name:
                    print(entry.name)
                    self.fileDict[entry.name.split("-")[3].rstrip(".tif")] = filelocation+entry.name



        print(self.sampleName)

        #for thing in listofString:

            #self.fileDict[thing.replace("\\","/").split("/")[-1].split("-")[1].rstrip(".tif")] = thing.replace("\\","/")

            #self.ListFile2.append(thing.replace("\\","/"))
            #print(thing.replace("\\","/").split("/")[-1].split("-")[1].rstrip(".tif"))


    def parseFiles(self):
        #print(self.fileDict)

        try:
            os.mkdir(self.outloc)
        except:
            pass


        values = []
        sortedKeys = sorted(self.fileDict)
        j = 0
        for sample in sortedKeys:
            #print(sample)
            j = j +1

            if not j%4 == 0:
                print(j)
                continue
            image1 = imageAnalysis(self.fileDict[sample])
            image1.openImage()
            image1.segmentImage9()
            image1.detectBlobs1()
            #pi = Image.fromarray(na)
            imageSave = outloc2+str(sample)+".png"
            output = image1.runGraph(imageSave)
            #img_with_graph = output[0]

            values.append([sample,output[1],output[3]])
            #img_with_graph  = Image.open(image1.runGraph())


            #del img_with_graph
            del image1
            for i in output:
                del i
            print(gc.collect())

            #frames.append(image1.runGraph())
        data1 = pd.DataFrame(data = values, columns = ["Sample","Node Degree","Void"])
        data1.to_csv(self.outloc+str(self.sampleName)+".csv")
        #frame_one = frames[0]
        #frame_one.save("thing2.gif", format="GIF", append_images=frames,
                   #save_all=True, duration=100, loop=0)


image2 = imageParser(location,outloc2)
image2.parseFiles()
