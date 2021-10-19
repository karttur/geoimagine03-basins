'''
Created on 5 May 2021

@author: thomasgumbricht
'''

import os 
from sys import platform
import numpy as np

from scipy.spatial.distance import cdist

import geoimagine.support.karttur_dt as mj_dt 


class ProcessBasins():
    '''class for binding to GRASS commands''' 
      
    def __init__(self, pp, session):
        '''
        '''
        
        self.session = session
                
        self.pp = pp  
        
        self.verbose = self.pp.process.verbose 
        
        self.session._SetVerbosity(self.verbose) 
              
        # Direct to subprocess
        if self.pp.process.processid == 'BasinOutletTiles':
                        
            self._BasinOutletTiles()
            
        else:
            
            SNULLE
                   
    def _BasinOutletTiles(self):
        '''
        '''
        
        if self.pp.process.parameters.asscript:
            
            today = mj_dt.Today()
            
            if platform == 'darwin':
                
                drivesys = '/Volumes'
                            
            scriptFP = os.path.join(drivesys, self.pp.dstPath.volume, self.pp.procsys.dstsystem, self.pp.dstCompD['outlet-pt'].source, 'region', 'basin-outlet-script')
            
            if not os.path.exists(scriptFP):
                
                os.makedirs(scriptFP)
                
            scriptFN = 'basin-corrected-watershed-%(today)s.sh' % {'today':today}
            
            self.scriptFPN = os.path.join(scriptFP, scriptFN)
            
            self.scriptF = open(self.scriptFPN, 'w')
            
            cmd = '# Script created by Kartturs GeoImagine Framework for refining basin-outlet-pt, created %s\n\n' % (today)
            
            cmd += '# Basin delineation: Create virtual DEM for directing hydrological flow in river basin mouths with GRASS\n\n'

            cmd += '# To run this script you must have GRASS GIS open in a Terminal window session.\n'
            cmd += '# Change the script to be executable by the command:\n'
            cmd += '# chmod 755 %(fpn)s\n' % {'fpn': self.scriptFPN}
            cmd += '# Then execute the command from your GRASS terminal session:\n'
            cmd += '# GRASS 7.x.x ("region"):~ > %(fpn)s\n\n' % {'fpn': self.scriptFPN}  
                    
            self.scriptF.write(cmd)
           
        for locus in self.pp.srcLayerD:
            
            self.tile = locus
            
            if self.verbose:
                
                infostr = '\n        %s' % (locus)
            
                print (infostr)
            
            for datum in self.pp.srcPeriod.datumL:
              
                allDone = True
                
                skip = False
                
                for dstComp in self.pp.dstCompL:
                
                    if not self.pp.dstLayerD[locus][datum][dstComp]._Exists() or self.pp.process.overwrite:
                                            
                        allDone = False
                                                       
                # if not allDone:
                    
                self.dstOutletPtsVectorLayer = self.pp.dstLayerD[locus][datum]['outlet-pt']
                
                self.dstFlowDirRasterLayer = self.pp.dstLayerD[locus][datum]['dem']
                
                for srcComp in self.pp.srcCompL:
                    
                    self.iniOutletPtsVectorLayer = self.pp.srcLayerD[locus][datum]['outlet-pt']
                    
                    self.shorewallPtsVectorLayer = self.pp.srcLayerD[locus][datum]['shorewall-pt']
                    
                    # THIS IS ALREADY DONE
                    if not self.pp.srcLayerD[locus][datum][srcComp]._Exists():
                                                    
                        skip = True
                        
                if allDone:
                    
                    continue
                            
                if skip:
                    
                    # only export the old flowdir MFDflowdir
                    
                    self._ExportFlowdir()
                    
                else:

                    self._Stage4()
                                            
        if self.pp.process.parameters.asscript:
            
            self.scriptF.close()
            
            infostr = 'Please inspect and run the script file %s' % (self.scriptFPN)
            
            print (infostr)
          
        # Repeat, but for running r.water.outlet 
        
        today = mj_dt.Today()
            
        if platform == 'darwin':
            
            drivesys = '/Volumes'
                        
        scriptFP = os.path.join(drivesys, self.pp.dstPath.volume, self.pp.procsys.dstsystem, self.pp.dstCompD['outlet-pt'].source, 'region', 'basin-outlet-script')
        
        if not os.path.exists(scriptFP):
            
            os.makedirs(scriptFP)
            
        scriptFN = 'grass_r-water-oulet_%(today)s.sh' % {'today':today}
        
        scriptFPN = os.path.join(scriptFP, scriptFN)
        
        GRASSbasinshF = open(scriptFPN, 'w')
        
        self.pp.process.parameters.asscript = True
          
        for locus in self.pp.srcLayerD:
            
            self.tile = locus
                        
            for datum in self.pp.srcPeriod.datumL:
                
                skip = False
                
                self.dstOutletPtsVectorLayer = self.pp.dstLayerD[locus][datum]['outlet-pt']
                
                self.dstFlowDirRasterLayer = self.pp.dstLayerD[locus][datum]['dem']
                
                for srcComp in self.pp.srcCompL:
                    
                    self.iniOutletPtsVectorLayer = self.pp.srcLayerD[locus][datum]['outlet-pt']
                    
                    self.shorewallPtsVectorLayer = self.pp.srcLayerD[locus][datum]['shorewall-pt']
                    
                    # THIS IS ALREADY DONE
                    if not self.pp.srcLayerD[locus][datum][srcComp]._Exists():
                                                    
                        skip = True
                                  
                if skip:
                    
                    continue
                    
                else:
        
                    outletsonly = True
                    
                    self._Stage4(outletsonly)
                    
                    tilescriptFPN = self._DelineateBasin(locus)
                    
                    cmd = 'chmod 755 %s\n' % (tilescriptFPN)
                    
                    cmd += '%s\n\n' % (tilescriptFPN) 
                    
                    GRASSbasinshF.write(cmd)
                    
        GRASSbasinshF.close()
            
        infostr = 'Please inspect and run the script file %s' % (scriptFPN)
            
        print (infostr)
         
    def _SetPtfieldD(self):
        ''' Set the fields for point data
        '''

        fieldDD = {}

        fieldDD['tile'] = {'name':'tile', 'type':'string', 'width':6,
                           'precision':0, 'transfer':'constant'}
        
        fieldDD['UPSTREAM'] = {'name':'upstream', 'type':'real', 'width':24,
                              'precision':8, 'transfer':'constant' }

        fieldDD['mouth_id'] = {'name':'mouth_id', 'type':'string', 'width':16,
                               'precision':0, 'transfer':'constant'}

        fieldDD['basin_id'] = {'name':'basin_id', 'type':'string', 'width':16,
                                'precision':0, 'transfer':'constant' }

        fieldDD['x'] = {'name':'x', 'type':'real', 'width':24,
                               'precision':8, 'transfer':'constant' }

        fieldDD['y'] = {'name':'y', 'type':'real', 'width':24,
                                'precision':8, 'transfer':'constant' }

        fieldDD['x_orig'] = {'name':'x_orig', 'type':'real', 'width':24,
                               'precision':8, 'transfer':'constant' }

        fieldDD['y_orig'] = {'name':'y_orig', 'type':'real', 'width':24,
                                'precision':8, 'transfer':'constant' }

        return fieldDD
                                            
    def _Stage4(self, outletsonly=False):
        '''
        '''

        if self.pp.process.parameters.outlet.upper() == 'MOUTH':
            
            if self.pp.process.parameters.distill.upper() == 'MOUTH':
                
                if self.verbose:
                    
                    infostr = '        Identifying outlets from full width mouth data distilled using mouth and basin clusters\n'
                    
                    infostr += '        outlet candidates: %s' % (self.iniOutletPtsVectorLayer.FPN)
                    
                    print (infostr)

                self.MouthMouthOutlets()

            else:
                
                # copies all input mouth cells to become basin outlet points
                if self.verbose:
                    
                    infostr = '        Outlets set equal to full width mouth data'
                    
                    print (infostr)
                    
                self.CopyDs()

        elif self.pp.process.parameters.outlet == 'SFD':
            
            if self.pp.process.parameters.distill == 'MFD':
                
                if self.verbose:
                    
                    infostr = '        Identifying outlets from SFD data distilled using MFD clusters'
                    
                    print (infostr)

                # finalOutlets, removedOutlets, spatialRef = SFDMFDoutlets(SFDsrcfpn, MFDsrcfpn, verbose, paramsD)

            else:
                
                # This is alternative is simply using the existing SFD
                
                if self.verbose:
                    
                    infostr = '        Outlets set equal to SFD input data'
                    
                    print (infostr)
                # finalOutlets, spatialRef = CopyDs(SFDsrcfpn, paramsD['@proj4CRS'])

        elif self.pp.process.parameters.outlet == 'MFD':
            
            if self.pp.process.parameters.distill == 'MFD':
                
                if self.verbose:
                    
                    infostr = '        Identifying outlets by distilling MFD clusters to unique outlets'
                    
                    print (infostr)
                    
                # finalOutlets, removedOutlets, spatialRef = MFDoutlets(MFDsrcfpn, verbose, paramsD)
                
            else:
                
                if self.verbose:
                    
                    infostr = '        Outlets set equal to MFD input data'
                    
                    print (infostr)
                    
                # This is alternative is simply using the existing MFD
                # finalOutlets, spatialRef = CopyDs(MFDsrcfpn, paramsD['@proj4CRS'])
        else:
            exitstr = 'EXITING, the parameter outlet must be set to either mouth, MFD or SFD'
            exit(exitstr)

        if self.verbose:
            
            print ('        Total nr of final outlets', len(self.finalOutletsD))

        fieldDefD = self._SetPtfieldD()
            
        self.iniOutletPtsVectorLayer.CreateVectorAttributeDef(fieldDefD)
        
        fieldDefL = self.iniOutletPtsVectorLayer.fieldDefL
        
        spatialRef = self.iniOutletPtsVectorLayer.layer.spatialRef
        
        # conert the finalOutletsD dict to list
        finalOutletsL = []
        
        for key in self.finalOutletsD:
            
            self.finalOutletsD[key]['tile'] = self.tile
            
            finalOutletsL.append(self.finalOutletsD[key])
        
        self.dstOutletPtsVectorLayer._VectorsCreateDsLayer(spatialRef, 'point', 'outlets', fieldDefL)
        
        self.dstOutletPtsVectorLayer._AddPtDataFromDict('x', 'y', finalOutletsL)
        
        if outletsonly:
            
            return

        if self.pp.process.parameters.outlet.upper() == 'MOUTH':
            
            if self.pp.process.parameters.tiltmouths:
                                            
                self.GRASSDEM()
                           
            else:
                        
                self.GRASSDEMnoTilt()

    def _DelineateBasin(self, locus):
        
        individualBasinFP = os.path.join(self.dstOutletPtsVectorLayer.FP, 'individual-basins')
        
        if not os.path.exists(individualBasinFP):
            
            os.makedirs(individualBasinFP)
            
        today = mj_dt.Today()
                                    
        scriptFP = individualBasinFP
          
        scriptFN = '%(locus)s_grass_r-water-oulet_%(today)s.sh' % {'locus':locus, 'today':today}
        
        scriptFPN = os.path.join(scriptFP, scriptFN)
        
        basinFN = 'basins_dem_%s_0_all.shp' % (locus)
        
        basinFPN = os.path.join(scriptFP, basinFN)
        
        GRASStileBasinshF = open(scriptFPN, 'w')
    
        for o in self.finalOutletsD:

            if o < 10:
                oStr = '%s_00000%s' % (locus, o)
            elif o < 100:
                oStr = '%s_0000%s' % (locus, o)
            elif o < 1000:
                oStr = '%s_000%s' % (locus, o)
            elif o < 10000:
                oStr = '%s_00%s' % (locus, o)
            elif o < 100000:
                oStr = '%s_0%s' % (locus, o)
            else:
                oStr = '%s_%s' % (locus, o)

            layername = '%(typ)s_%(d)s' % {'typ': self.pp.process.parameters.outlet, 'd':oStr}

            # Set ESRI shape filename
            shpfn = '%sc.shp' % (layername)

            shpfpn = os.path.join(individualBasinFP, shpfn)

            if os.path.exists(shpfpn) and not self.pp.process.overwrite:
                continue

            xcoord = self.finalOutletsD[o]['x']

            ycoord = self.finalOutletsD[o]['y']

            mouth_id = self.finalOutletsD[o]['mouth_id']

            basin_id = self.finalOutletsD[o]['basin_id']
            
            cmd = 'g.mapset mapset=%(locus)s\n' % {'locus': locus}
            
            cmd += 'g.region raster=%s\n' % (self.pp.process.parameters.grassdem)
            
            cmd += 'r.water.outlet input=%(draindir)s output=%(ln)s coordinates=%(x)f,%(y)f --overwrite\n' % {'draindir': self.wsheddraindir, 'x': xcoord, 'y':ycoord, 'ln':layername}

            # cmd += '# in 7.8 NULL is set in r.water.outlet # r.null %(ln)s setnull=0\n' %{'ln': layername}

            cmd += 'r.to.vect input=%(ln)s output=%(ln)s type=area --overwrite\n' % {'ln': layername}

            cmd += 'g.remove -f type=raster name=%(ln)s --quiet\n' % {'ln': layername}

            cmd += 'v.clean input=%(ln)s output=%(ln)sc type=area tool=prune,rmdupl,rmbridge,rmline,rmdangle thresh=0,0,0,0,-1 --overwrite\n' % {'ln': layername}

            cmd += 'v.db.addcolumn map=%(ln)sc columns="x_mouth DOUBLE PRECISION, y_mouth DOUBLE PRECISION, tile varchar(16)"\n' % {'ln': layername}

            cmd += 'v.db.update map=%(ln)sc column=x_mouth value=%(x)f\n' % {'ln': layername, 'x':xcoord}

            cmd += 'v.db.update map=%(ln)sc column=y_mouth  value=%(y)f\n' % {'ln': layername, 'y':ycoord}

            cmd += 'v.db.update map=%(ln)sc column=tile  value=%(locus)s\n' % {'ln': layername, 'locus':locus}

            # if self.params.process.parameters.stage4_method == 'allmouths':
            if True:

                cmd += 'v.db.addcolumn map=%(ln)sc columns="mouth_id INT, basin_id INT"\n' % {'ln': layername}

                cmd += 'v.db.update map=%(ln)sc column=mouth_id  value=%(mouth)d\n' % {'ln': layername, 'mouth':mouth_id}

                cmd += 'v.db.update map=%(ln)sc column=basin_id  value=%(basin)d\n' % {'ln': layername, 'basin':basin_id}

            cmd += 'v.to.db map=%(ln)sc type=centroid option=area columns=area_km2 units=kilometers\n' % {'ln': layername}

            cmd += 'v.out.ogr input=%(ln)sc type=area format=ESRI_Shapefile output=%(dsn)s --overwrite\n\n' % {'ln': layername, 'dsn':shpfpn}

            GRASStileBasinshF.write(cmd)
            
        # 
        
        searchstr = '%s_%s_*c.shp' % (self.pp.process.parameters.outlet, locus)
        
        searchpath = os.path.join(individualBasinFP,searchstr)

        cmd = '# Shell script loop for joining all identified river basins to a single vector file\n'

        #cmd += 'cd %(fp)s\n\n' % {'fp':individualBasinFP}

        cmd += 'files=(%(search)s)\n\n' % {'search':searchpath}

        cmd += 'first=${files[0]}\n\n'

        cmd += 'files=("${files[@]:1}")\n\n'

        cmd += 'ogr2ogr -skipfailures %(dstFPN)s "$first"\n\n' % {'dstFPN': basinFPN}

        cmd += 'for file in "${files[@]}"; do\n'

        cmd += '    echo "$file"\n'

        cmd += '    ogr2ogr -append -skipfailures %(dstFPN)s "$file"\n\n' % {'dstFPN': basinFPN}

        cmd += 'done\n'

        # shFN = '%(s)s_ogr2ogr-patch-basins_%(typ)s_stage4-step3B.sh' %{'s':self.params.locus, 'typ':self.params.process.parameters.stage4_method}

        GRASStileBasinshF.write(cmd)
            
        GRASStileBasinshF.close()
            
        return (scriptFPN)
            
    def MouthMouthOutlets(self):
        ''' Distill one outlet point per full width mouth
        '''

        # MouthMouthOutlets does not include remo
        self.removedOutlets = 0

        self.iniOutletPtsVectorLayer._VectorOpenGetFirstLayer()
        
        self.shorewallPtsVectorLayer._VectorOpenGetFirstLayer()
        
        # srcLayer = ds.GetLayer()

        # Open the shore wall points

        # swds = self.OpenDs(self.params.FPNs.shorewallptfpn_s2)

        # swLayer = swds.GetLayer()

        # self.GetSpatialRef(srcLayer)

        self.featureCount = self.iniOutletPtsVectorLayer.layer.layer.GetFeatureCount()
        
        if self.verbose:
            
            print ("        Number of candidate outlets: %d" % (self.featureCount))

        self.UniqueDbPt(self.iniOutletPtsVectorLayer.layer.layer, self.shorewallPtsVectorLayer.layer.layer)

    def SFDMFDoutlets(self, SFDsrcfpn, MFDsrcfpn):
        ''' Distill SFD outlets from MFD clusters
        '''

        if self.verbose:
            infostr = '    Cleaning candidate SFD outlets from clustered MFD outlets'
            print (infostr)

        # Open the SFD DS and get the point layer
        SFDsrclayer = self.OpenDs(SFDsrcfpn)

        # Open the MFD DS and get the point layer
        # MFDsrclayer = self.OpenDs(MFDsrcfpn)

        # Get all outlet points that are clustered in the MFD layer
        clusteredD = self.ClusteredOutlets()

        # get the complete feature data for the SFD outlets
        self.featsL = [(feature.GetGeometryRef().GetX(), feature.GetGeometryRef().GetY(), feature.GetField("upstream"),
                   feature.GetField("xcoord"), feature.GetField("ycoord"), feature.GetField("basin")) for feature in SFDsrclayer]

        if self.verbose:
            infostr = '    Initial number of SFD outlets: %s' % (len(self.featsL))
            print (infostr)

        # For each cluster, get all the SFD within the cluster, and only retain the one with the largest upstream
        SFDremoveL = []
        for cluster in clusteredD:
            SFDclusterL = []
            for cell in clusteredD[cluster]:
                for x, slot in enumerate(self.featsL):
                    if cell[0] == slot[0] and cell[1] == slot[1]:
                        SFDclusterL.append((x, slot[0], slot[1], slot[2], slot[3]))
            if len(SFDclusterL) > 1:
                clusterL = sorted(SFDclusterL, key=itemgetter(3), reverse=True)

                for i, item in enumerate(clusterL):

                    if i > 0:
                        SFDremoveL.append(item[0])

        SFDremoveL.sort(reverse=True)

        self.removeL = [self.featsL.pop(x) for x in SFDremoveL]

        if self.verbose:
            infostr = '    Removing %s SFD outlets identified as clustered from MFD outlets' % (len(SFDremoveL))
            print (infostr)

    def MFDoutlets(self):

        srcLayer = self.OpenDs(self.MFDsrcfpn)

        # Get the closest point of each candidate basin outlet as a list of np.array [x y]
        self.finalOutletL, self.removedOutlets = self.UniqueOutlets(srcLayer)

        # Save and close everything
        ds = layer = feat = geom = None

    def UniqueDbPt(self, srcLayer, swLayer):
        '''
        '''

        # get the feature data for the input layer
            
        featsL = [(feature.GetGeometryRef().GetX(),
                        feature.GetGeometryRef().GetY(),
                        feature.GetField("upstream"),
                        feature.GetField('mouth_id'),
                        feature.GetField('basin_id')) for feature in srcLayer]

        featsParamL = ['x', 'y', 'upstream', 'mouth_id', 'basin_id']

        self.featsL = []

        for item in featsL:
            
            self.featsL.append(dict(zip(featsParamL, item)))

        # sw = shore wall
        swCoordA = np.asarray([ [feature.GetGeometryRef().GetX(), feature.GetGeometryRef().GetY() ] for feature in swLayer])

        mouthD = {}
        
        outletD = {}
        
        self.widestMouth = 0
        
        for f in self.featsL:

            if not f['mouth_id'] in mouthD:
                
                mouthD[f['mouth_id']] = []
                
            mouthD[f['mouth_id']].append(f)

        if self.verbose:

            print ('        identified nr of mouths', len(mouthD))
                    
        for m in mouthD:
            
            if len(mouthD[m]) == 1:
                
                # if a single mouth, no conflict, just add
                outletD[m] = mouthD[m][0]
                
            else:

                # Multiple outlets for one mouth
                if len(mouthD[m]) > self.widestMouth:
                    
                    # Identiry the widest mouth
                    self.widestMouth = len(mouthD[m])

                if self.pp.process.parameters.clusteroutlet[0].lower() == 'c':
                    
                    # Construct a numpy array of coords
                    coordA = np.asarray([ [p[0], p[1]] for p in mouthD[m] ])
                    
                    print (coordA)
                    SNULLE
                    
                    # Calculate the geometrical most central outlet point
                    avgXY = np.average(coordA, axis=0)

                    # Find the mouth point closest to the average
                    c = cdist([avgXY], coordA).argmin()

                    # Set the outlet to the point closest to the average
                    outletD[m] = mouthD[m][c]
                    
                else:
                    
                    u = 0
                    
                    for p in mouthD[m]:

                        if p['upstream'] > u:
                            
                            u = p['upstream']
                            
                            outletD[m] = p

        if self.verbose:
            
            print ('        Identified %s outlet points' % (len(outletD)))
            
            print ('        Widest mouth = %s pixels' % (self.widestMouth))
   
        # Find the point in the wall that is closest to the original outlet, and move the point
        
        swD = {}
        
        for o in outletD:

            # Only retain candidates that are above the minimum basin area in km2 (minBasinAreaKm2)

            if outletD[o]['upstream'] < self.pp.process.parameters.basincelltrehshold:

                continue

            # set the outlet as a numpy coordinate
            oa = np.asarray([ [outletD[o]['x'], outletD[o]['y']] ])

            n = cdist(oa, swCoordA).argmin()

            swD[o] = {'x':swCoordA[n][0],
                      'y':swCoordA[n][1],
                      'upstream':outletD[o]['upstream'],
                      'mouth_id':outletD[o]['mouth_id'],
                      'basin_id':outletD[o]['basin_id'],
                      'x_orig':outletD[o]['x'],
                      'y_orig':outletD[o]['y'] }

        self.finalOutletsD = swD
        self.dstremfpn = self.removedOutletsD = 0

    def ClusterAdjacent(self):
        ''' Cluster adjacent cells
        '''

        # Convert the list of coordinates to an array
        adjA = np.array(self.adjacentL)

        # Create a Dict to hold the clusters of candidate basin outlet cells
        self.clusterD = {}

        # Create a counter for the Dict clusterD
        nrKeys = 0

        # Get the total number of candidate basin outlets
        totalAdjCells = adjA.shape[0]

        if self.verbose:
            print ('    Total number of cells to cluster', totalAdjCells)

        # Loop all candidate nodes
        for i, node in enumerate(adjA):
            if self.verbose > 1:
                print ('    Processing cell nr', i)

            # Get all nodes, except the node itself
            nodes = np.delete(adjA, [i], 0)

            # Retrieve the cells that fall within dt as a Boolean array
            within = cdist([node], nodes) < self.pp.process.parameters.thresholddist

            # Mask the nodes (original coordinates) with the Boolean array
            adjacents = nodes[within[0], :]

            # Append the original node to the array of nodes adjacent to the source node of this loop
            aA = np.append([node], adjacents, axis=0)

            # Loop cluster Dicts to check if this cluster is already registered
            cellin = False
            for cell in aA:
                for key in self.clusterD:
                    for node in self.clusterD[key]:
                        # check if the cell and the node are identical
                        if cell[0] == node[0] and cell[1] == node[1]:
                            # if a single cell of this new cluster is already registerd, the whole cluster belongs to the existing cluster
                            cellin = True
                            theKey = key
                            break
            if cellin:  # This cluster is already in the Dict
                # Loop over the cells in the new cluster to check that they are actually in the Dict
                for cell in aA:
                    cellin = False
                    for item in self.clusterD[theKey]:
                        if cell[0] == item[0] and cell[1] == item[1]:
                            cellin = True
                    if not cellin:
                        # if the individual cell of this cluster is not registered, add it
                        self.clusterD[theKey] = np.append(self.clusterD[theKey], [cell], axis=0)

            else:
                # A new cluster is needed, add 1 to nrKeys
                nrKeys += 1
                theKey = nrKeys

                # Set Dict cluster to all of the cells in the cluster
                self.clusterD[nrKeys] = aA

        # End of for i,node in enumerate(adjA)

        # Calculate the number of clusters and the total number of nodes in those clusters
        if not self.CalcClusterContent(totalAdjCells):

            if self.verbose:
                print ('    Some nodes are registered as duplicates, cleaning clusters')

            # If there are duplicates within the clusters, these clusters need to be joined

            # Declare variables for holding clusters identied to have duplicates
            keyDelL = []
            keyLinkD = {}
            keyDoneL = []
            duplFound = 0

            # Loop over all the clusters
            for key in self.clusterD:

                # Inside the first loop, loop all clusters again
                for keyTwo in self.clusterD:

                    # If the outer and inner loop is for the same, continue
                    if key == keyTwo:
                        continue

                    # if the other loop key is already processed in the inner loop, continue
                    if key in keyDoneL:
                        continue

                    # Compare all the cells of the other loop with all the cells of the inner loop
                    for cell in self.clusterD[key]:
                        for item in self.clusterD[keyTwo]:
                            # If any cell in the outher and inner llop clusters is the same, note that
                            if cell[0] == item[0] and cell[1] == item[1]:
                                keyDoneL.append(keyTwo)
                                duplFound += 1
                                keyDelL.append(keyTwo)
                                if not key in keyLinkD:
                                    keyLinkD[key] = []
                                if not keyTwo in keyLinkD[key]:
                                    keyLinkD[key].append(keyTwo)

            if self.verbose:
                printstr = '    Identified %(d)d duplicate nodes, transferring and deleting' % {'d':duplFound}
                print (printstr)

            # Loop over the Dict listing clusters having duplicates
            for key in keyLinkD:
                # Loop over the duplicated clusters of the outer loop cluster
                for keyTwo in keyLinkD[key]:
                    # loop over the cells in the clusterD[keyTwo],
                    # if that cell is not in clusterD[key] - move it
                    for cell in self.clusterD[keyTwo]:
                        cellin = False
                        for item in self.clusterD[key]:
                            if cell[0] == item[0] and cell[1] == item[1]:
                                cellin = True
                        if not cellin:
                            self.clusterD[key] = np.append(self.clusterD[key], [cell], axis=0)

                # Delete the redundant cluster
                del self.clusterD[keyTwo]

            if not self.CalcClusterContent(totalAdjCells):
                exit('    Error in the number of clustered cells')

        # return clusterD

    def CentralClusterOutlet(self):
        ''' Identify the most geometrically central cell of each cluster
        '''

        # Declare Dict to hold the cluster cell closest to the cluster center
        clusterOutletD = {}

        for key in self.clusterD:
            avgXY = np.average(self.clusterD[key], axis=0)
            clusterOutletD[key] = self.clusterD[key][cdist([avgXY], self.clusterD[key]).argmin()]

        return clusterOutletD

    def LargestClusterOutlet(self, featsL):
        ''' Identify the cell with the highest accumulation
        '''

        # Declare Dict to hold the cluster cell with he largest upstream area
        clusterOutletD = {}

        for key in self.clusterD:
            maxupstream = 0
            for cell in self.clusterD[key]:
                for f in featsL:
                    if f[0] == cell[0] and f[1] == cell[1]:
                        if f[2] > maxupstream:
                            maxupstream = f[2]
                            clusterOutlet = cell

            clusterOutletD[key] = clusterOutlet

        return clusterOutletD

    def UniqueOutlets(self, layer, dt, clusterOutlet):
        ''' Separate basin outlets defined as single cells compared to multiple adjacent vells
        '''

        # get coordinates of points as 2d array
        coords = np.array([(feature.GetGeometryRef().GetX(), feature.GetGeometryRef().GetY()) for feature in layer])

        layer.ResetReading()

        # get the feature data as well
        # featsL = [(feature.GetGeometryRef().GetX(), feature.GetGeometryRef().GetY(), feature.GetField("upstream"), feature.GetField("basin")) for feature in layer]
        featsL = [(feature.GetGeometryRef().GetX(), feature.GetGeometryRef().GetY(), feature.GetField("upstream"),
                   feature.GetField("xcoord"), feature.GetField("ycoord"), feature.GetField("basin")) for feature in layer]
        layer.ResetReading()

        # Create two lists, holding uniquely separated outlets and those that are adjacent
        adjacentL = []  # list of basin outlets that are at close proximity

        uniqueL = []  # list of single cell basin outlets

        # Calculate distance matrix between points
        for i, pt in enumerate(coords):

            self.ClosestNode(pt, np.delete(coords, [i], 0))

        self.ClusterAdjacent()

        if clusterOutlet.lower()[0] == 'c':
            clusterOutletD = self.CentralClusterOutlet()
            if self.verbose:
                print ('    Looking for central outlet in clusters')
        else:
            clusterOutletD = self.LargestClusterOutlet(featsL)
            if self.verbose:
                print ('    Looking for maximum outlet in clusters')

        # Join the clusterMouths to the uniqueL
        for key in clusterOutletD:
            uniqueL.append(clusterOutletD[key])

        # Restore the original order of outlets, and add the upstream area
        self.finaloutletL = []
        self.removedOutlets = []

        for candpt in featsL:
            candin = False
            for outlet in uniqueL:
                if outlet[0] == candpt[0] and outlet[1] == candpt[1]:
                    self.finaloutletL.append(candpt)
                    candin = True
            if not candin:
                self.removedOutlets.append(candpt)

    def ClusteredOutlets(self, layer):
        ''' Separate basin outlets defined as single cells compared to multiple adjacent vells
        '''

        # get coordinates of points as 2d array
        coords = np.array([(feature.GetGeometryRef().GetX(), feature.GetGeometryRef().GetY()) for feature in layer])

        layer.ResetReading()

        # Create two lists, holding uniquely separated outlets and those that are adjacent
        self.adjacentL = []  # list of basin outlets that are at close proximity

        self.uniqueL = []  # list of single cell basin outlets

        # Calculate distance matrix between points
        for i, pt in enumerate(coords):

            self.ClosestNode(pt, np.delete(coords, [i], 0))

        clusterD = self.ClusterAdjacent()

        return clusterD

    def GetDistilled(self, dstfpn, remdstfpn):

        dstlayer = self.OpenDs(dstfpn)

        # Copy all features to finalOutletL
        self.finalOutletL = [(feature.GetGeometryRef().GetX(), feature.GetGeometryRef().GetY(), feature.GetField("upstream")) for feature in dstlayer]

        self.removedOutletL = False

        if remdstfpn:
            remdstlayer = self.OpenDs(remdstfpn)
            self.removedOutletL = [(feature.GetGeometryRef().GetX(), feature.GetGeometryRef().GetY(), feature.GetField("upstream")) for feature in remdstlayer]

    def ClosestNode(self, node, nodes):
        ''' Separate nodes based on the closest distance to any other node
        '''

        argmindist = cdist([node], nodes).argmin()
        mindist = cdist([node], nodes)[0, argmindist]

        if mindist > self.params['threholddist']:
            # The distance to the nearest other point is longer than the given threshold
            self.uniqueL.append(node)

        else:
            # The distance to the nearest other point is shorter than or equal to the given threshold
            self.adjacentL.append(node)

    def CalcClusterContent(self, totalAdjCells, clusterD, verbose):
        ''' Compare the nr of clusters and then nr of basin starting points
        '''

        totalCellsInClusters = 0
        nClusters = 0

        for key in clusterD:
            nClusters += 1
            totalCellsInClusters += clusterD[key].shape[0]
        if verbose:
            print ('    Total nr of cells in cluster', totalCellsInClusters)
            print ('    Total number of cells to cluster', totalAdjCells)

        return totalCellsInClusters == totalAdjCells

    def GetSpatialRef(self, layer):
        '''
        '''

        self.spatialRef = layer.GetSpatialRef()

    def GRASSDEM(self):
        ''' Creates script for virtual, hydrological corrected DEM with river mouths sloping towards single outlet point
        '''

        # Calculate the max distance for the cost grow
        if self.pp.process.parameters.clusteroutlet[0].lower() == 'c':
            d = self.widestMouth * 0.71
        else:
            d = self.widestMouth

        # Set a baseprefix
        grassbaseprefex = 'basin_%s_%s' % ('distill', self.pp.process.parameters.distill)

        cmd = '# change to the mapset for this tile \n'
        
        cmd += 'g.mapset %s\n\n' % (self.tile)
        
        cmd += '# set region to full dem size \n'
        
        cmd += 'g.region raster=%s\n\n' % (self.pp.process.parameters.grassdem)
        
        cmd += '# Import outlet points vector created earlier [steage 2] \n'

        swptv = '%s_%s' % (grassbaseprefex, 'shorewall_pt')

        cmd += 'v.in.ogr -o input=%(in)s output=%(out)s --overwrite\n\n' % {'in':self.dstOutletPtsVectorLayer.FPN, 'out':swptv}

        swptr = '%s_%s' % (grassbaseprefex, 'shorewall')

        cmd += '# Rasterize the outlet points\n'

        cmd += 'v.to.rast input=%(in)s output=%(out)s type=point use=val value=%(d)d --overwrite\n\n' % {'in':swptv, 'out':swptr, 'd':1 }

        cmd += ('# add filled holes to the shoredem\n')

        swfillr = '%s_%s' % (grassbaseprefex, 'shorewall_fill')

        cmd += 'r.mapcalc "%(swfillr)s = if(isnull(%(swptr)s),fillholeDEM, %(swptr)s)" --overwrite\n\n' % {'swfillr':swfillr, 'swptr':swptr}

        cmd += '# Add filled outlet points to river mouth DEM with all cells=1\n'

        lowDEMp1 = '%s_%s' % (grassbaseprefex, 'lowlevel_DEM')

        cmd += 'r.mapcalc "%(lowdem)s = if(isnull(%(swfillr)s),lowlevel_outlet_costgrow+1,%(swfillr)s)" --overwrite\n\n' % {'lowdem':lowDEMp1, 'swfillr':swfillr}

        cmd += '# cost grow analysis from new outlets over river mouth DEM\n'

        costDEM = '%s_%s' % (grassbaseprefex, 'mouth_dist')

        cmd += 'r.cost -n input=%(lowdem)s output=%(out)s start_points=%(pt)s max_cost=%(d)d\
            memory=%(mem)d --overwrite\n\n' % {'lowdem':lowDEMp1, 'out':  costDEM, 'pt': swptv, 'd':d, 'mem':self.pp.process.parameters.memory}

        # cmd += '# export cost grow analysis (optional)\n'

        # cmd += '# r.out.gdal -f input=%(in)s format=GTiff type=Int16 output=%(out)s --overwrite\n\n' %{'in':costDEM, 'out':self.params.FPNs.BasinMouthCostGrowFpn_s4}

        cmd += '# Invert cost grow to create mouth flow route DEM directing flow towards outlet point\n'

        routedem = '%s_%s' % (grassbaseprefex, 'routing_dem')

        cmd += 'r.mapcalc "%(out)s = int(%(in)s-%(d)d)" --overwrite\n\n' % {'out':routedem, 'in':costDEM, 'd':d + 2}

        # cmd += '# export the mouth flow route DEM (optional)\n'

        # cmd += '# r.out.gdal -f input=%(in)s format=GTiff type=Int16 output=%(out)s --overwrite\n\n' %{'in':routedem, 'out':self.params.FPNs.BasinMouthRouteDEMFpn_s4}

        cmd += '# combine mouth flow route DEM with original DEM and add the shorewall\n'

        hydrodem = '%s_%s' % (grassbaseprefex, 'basin_dem')

        cmd += 'r.mapcalc "%(hydrodem)s = if(isnull(%(routedem)s),(if(isnull(%(dem)s),%(shorewall)s,%(dem)s)),%(routedem)s)" --overwrite\n\n' % {'hydrodem':hydrodem, 'shorewall':'shorewall', 'dem':self.pp.process.parameters.grassdem, 'routedem':routedem}

        # cmd += '# export the hydrological corrected DEM (optional)\n'

        # cmd += '# r.out.gdal -f input=%(hydrodem)s format=GTiff type=Int16 output=%(out)s --overwrite\n\n' %{'hydrodem':hydrodem, 'out':self.params.FPNs.BasinHydroDEMFpn_s4}

        cmd += '# run r.watershed for the new outlet points and the virtual (hydrologically corrected) DEM\n'

        wsprefix = '%s_%s' % (grassbaseprefex, 'wshed')

        self.wshedacc = '%(wsp)s_acc' % {'wsp':wsprefix}

        self.wsheddraindir = '%(wsp)s_draindir' % {'wsp':wsprefix}

        if self.pp.process.parameters.watershed == 'SFD':

            cmd += 'r.watershed -as convergence=%(cnv)d elevation=%(hydrodem)s accumulation=%(acc)s drainage=%(drain)s_draindir threshold=%(threshold)d --overwrite\n\n' % {'hydrodem':hydrodem, 'acc':self.wshedacc, 'drain':self.wsheddraindir}

        elif self.pp.process.parameters.watershed == 'MFD':

            cmd += 'r.watershed -a elevation=%(hydrodem)s accumulation=%(acc)s drainage=%(drain)s threshold=%(threshold)d\n\n' % {'hydrodem':hydrodem,
                    'acc':self.wshedacc, 'drain':self.wsheddraindir, 'threshold':self.pp.process.parameters.basincelltrehshold}

        else:

            convergence = int(self.pp.process.parameters.watershed[3])

            cmd += 'r.watershed -a convergence=%(cnv)d elevation=%(hydrodem)s accumulation=%(acc)s drainage=%(drain)s threshold=%(threshold)d --overwrite\n\n' % {'cnv':convergence, 'hydrodem':hydrodem, 'acc':self.wshedacc, 'drain':self.wsheddraindir}

        cmd += '# Set region to core tile before exporting flowdir\n'

        cmd += 'g.region raster=originalTile\n\n'
        
        cmd += '# export the drainage (flowdir) as this is the basis for finding basins in later stages\n'

        cmd += 'r.out.gdal -f input=%(drain)s format=GTiff output=%(out)s --overwrite\n\n' % {'drain':self.wsheddraindir, 'out':self.dstFlowDirRasterLayer.FPN}

        # cmd += '# convert updrain accumulation raster to byte format (optional)\n'

        # wsaccln = '%(wsp)s_acc_ln' %{'wsp':wsprefix}

        # cmd += '# r.mapcalc "%(wsal)s = 10*log(%(acc)s)" --overwrite\n\n' %{'wsal':wsaccln, 'acc':self.wshedacc}

        # cmd += '# set color ramp for upstream accumulation raster (optional)\n'

        # cmd += '# r.colors map=%(wsal)s color=ryb\n\n' %{'wsal':wsaccln}

        # cmd += '# export the visualised upstream accumulation raster (optional)\n'

        # cmd += '# r.out.gdal -f input=%(wsal)s format=GTiff type=Byte output=%(out)s --overwrite\n\n' %{'wsal':wsaccln, 'out':self.params.FPNs.watershedUpdrainFpn_s4}

        # cmd += '# add column "updrain" to the new outlet vector\n'

        # cmd += 'v.db.addcolumn map=%(swptv)s columns="updrain DOUBLE PRECISION" \n\n' %{'swptv':swptv,'out':swptv}

        # cmd += '# extract data from r.watershed updrain to the column "updrain" in the outlet point map\n'

        # cmd += 'v.what.rast map=%(swptv)s column=updrain raster=%(acc)s\n\n' %{'swptv': swptv,'acc':self.wshedacc}

        # Write the commands to file

        # GRASS1shF = open(self.GRASS1shFPN,'w')

        self.scriptF.write(cmd)
        
    def GRASSDEMnoTilt(self):
        ''' Creates script for virtual, hydrological corrected DEM with river mouths sloping towards single outlet point
        '''

        # Calculate the max distance for the cost grow
        if self.pp.process.parameters.clusteroutlet[0].lower() == 'c':
            d = self.widestMouth * 0.71
        else:
            d = self.widestMouth

        # Set a baseprefix
        grassbaseprefex = 'basin_%s_%s' % ('distill', self.pp.process.parameters.distill)

        cmd = '# change to the mapset for this tile \n'
        
        cmd += 'g.mapset %s\n\n' % (self.tile)
        
        cmd += '# set region to full dem size \n'
        
        cmd += 'g.region raster=%s\n\n' % (self.pp.process.parameters.grassdem)
        
        cmd += '# Import outlet points vector created earlier [stage 2] \n'

        outptv = '%s_%s' % (grassbaseprefex, 'single_outlet_pt')

        cmd += 'v.in.ogr -o input=%(in)s output=%(out)s --overwrite\n\n' % {'in':self.dstOutletPtsVectorLayer.FPN, 'out':outptv }

        outptr = '%s_%s' % (grassbaseprefex, 'single_outlet')

        cmd += '# Rasterize the outlet points\n'

        cmd += 'v.to.rast input=%(in)s output=%(out)s type=point use=val value=%(d)d --overwrite\n\n' % {'in':outptv, 'out':outptr, 'd':-888 }
        
        vsr = '%s_%s' % (grassbaseprefex, 'virtual_hydro_shoreline')
        
        cmd += '# Overaly outlet points and thickwall\n'
        
        cmd += 'r.mapcalc "%(vsr)s = if(isnull(%(outptr)s),thickwall*9999, %(outptr)s)" --overwrite\n\n' % {'vsr':vsr, 'outptr':outptr }

        cmd += '# Overaly virtual shoreline DEM and the hydrologically inland corrected DEM\n'
        
        hydrodem = '%s_%s' % (grassbaseprefex, 'virtual_hydro_DEM')
        
        cmd += 'r.mapcalc "%(hydrodem)s = if(isnull(%(vsr)s),inland_comp_DEM, %(vsr)s)" --overwrite\n\n' % {'vsr':vsr, 'hydrodem':hydrodem }

        cmd += '# run r.watershed for the new outlet points and the virtual (hydrologically corrected) DEM\n'

        wsprefix = '%s_%s' % (grassbaseprefex, 'wshed')

        self.wshedacc = '%(wsp)s_acc' % {'wsp':wsprefix}

        self.wsheddraindir = '%(wsp)s_draindir' % {'wsp':wsprefix}

        if self.pp.process.parameters.watershed == 'SFD':

            cmd += 'r.watershed -as convergence=%(cnv)d elevation=%(hydrodem)s accumulation=%(acc)s drainage=%(drain)s_draindir threshold=%(threshold)d --overwrite\n\n' % {'hydrodem':hydrodem, 'acc':self.wshedacc, 'drain':self.wsheddraindir}

        elif self.pp.process.parameters.watershed == 'MFD':

            cmd += 'r.watershed -a elevation=%(hydrodem)s accumulation=%(acc)s drainage=%(drain)s threshold=%(threshold)d --overwrite\n\n' % {'hydrodem':hydrodem,
                    'acc':self.wshedacc, 'drain':self.wsheddraindir, 'threshold':self.pp.process.parameters.basincelltrehshold}

        else:

            convergence = int(self.pp.process.parameters.watershed[3])

            cmd += 'r.watershed -a convergence=%(cnv)d elevation=%(hydrodem)s accumulation=%(acc)s drainage=%(drain)s threshold=%(threshold)d --overwrite\n\n' % {'cnv':convergence, 'hydrodem':hydrodem, 'acc':self.wshedacc, 'drain':self.wsheddraindir}

        cmd += '# Set region to core tile before exporting flowdir\n'

        cmd += 'g.region raster=originalTile\n\n'
        
        cmd += '# export the drainage (flowdir) as this is the basis for finding basins in later stages\n'

        cmd += 'r.out.gdal -f input=%(drain)s format=GTiff output=%(out)s --overwrite\n\n' % {'drain':self.wsheddraindir, 'out':self.dstFlowDirRasterLayer.FPN}

        cmd += '# set to full (mosaic) region\n'
        
        cmd += '# g.region raster=DEM\n\n'
        
        cmd += '# convert updrain accumulation raster to byte format (optional)\n'

        wsaccln = '%(wsp)s_acc_ln' % {'wsp':wsprefix}

        cmd += '# r.mapcalc "%(wsal)s = 10*log(%(acc)s)" --overwrite\n\n' % {'wsal':wsaccln, 'acc':self.wshedacc}

        cmd += '# set color ramp for upstream accumulation raster (optional)\n'

        cmd += '# r.colors map=%(wsal)s color=ryb\n\n' % {'wsal':wsaccln}
        
        cmd += '# export the visualised upstream accumulation raster (optional)\n'
        
        flowaccFPN = os.path.join(self.dstFlowDirRasterLayer.FP, 'flowacc_pleasing.tif')

        cmd += '# r.out.gdal -f input=%(wsal)s format=GTiff type=Byte output=%(out)s --overwrite\n\n' % {'wsal':wsaccln, 'out':flowaccFPN}

        self.scriptF.write(cmd)
        
    def _ExportFlowdir(self):
        
        ''' Creates script for export existing flowdir
        '''

        # Set a baseprefix
        grassbaseprefex = 'basin_%s_%s' % ('distill', self.pp.process.parameters.distill)

        cmd = '# change to the mapset for this tile \n'
        
        cmd += 'g.mapset %s\n\n' % (self.tile)

        cmd += '# Set region to core tile before exporting flowdir\n'

        cmd += 'g.region raster=originalTile\n\n'
        
        cmd += '# export the drainage (flowdir) as this is the basis for finding basins in later stages\n'

        cmd += 'r.out.gdal -f input=MFDflowdir format=GTiff output=%(out)s --overwrite\n\n' % {'out':self.dstFlowDirRasterLayer.FPN}

        self.scriptF.write(cmd)

    
