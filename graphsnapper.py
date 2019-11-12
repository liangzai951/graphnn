import numpy as np
import graphtools as gt
from graphtools import stdout
from graphtools import h5py
import pandas as pd
import sys, getopt, os
import time
from tqdm import tqdm, trange

# Fifth ring
# xmin = 116.1904 * long2km
# xmax = 116.583642 * long2km
# ymin = 39.758029 * lat2km
# ymax = 40.04453 * lat2km

global_start = time.time()

# Grab MPI components from graphtools
mpicomm = gt.comm
mpirank = gt.rank
mpisize = gt.size

arglen = len(sys.argv[1:])
arglist = sys.argv[1:]

runname = ""
runpath = "/home/walterms/traffic/graphnn/veldata/"
outpath = "/home/walterms/traffic/graphnn/nn_inputs/"

node_mindist = 0.5
node_maxdist = 2.0
maxnbr = 8
nvel = None
disable_tqdm = False

try:
    opts, args = getopt.getopt(arglist,"",["node_mindist=","node_maxdist=","maxnbr=",\
        "nvel=","runname=", "runpath=", "outpath=", "disable_tqdm="])
except:
    stdout("Error in opt retrieval...", mpirank)
    sys.exit(2)

for opt, arg in opts:
    if opt == "--node_mindist":
        node_mindist = float(arg)
    elif opt == "--node_maxdist":
        node_maxdist = float(arg)
    elif opt == "--maxnbr":
        maxnbr = int(arg)
    elif opt == "--nvel":
        nvel = int(arg)
    elif opt == "--runname":
        runname = arg
    elif opt == "--runpath":
        runpath = arg
    elif opt == "--outpath":
        outpath = arg
    elif opt == "--disable_tqdm":
        disable_tqdm = bool(arg)

if not runname:
    stdout("Please specify velocity filename", mpirank)

stdout("Processing velocity file "+runpath+runname+" for nn inputs", mpirank)
stdout("Saving input files to "+outpath, mpirank)

# Write params to output info
if mpirank==0:
    info_outname = outpath+runname+".info"
    info_out = open(info_outname,'w')
    info_out.write("run "+runpath+runname+"\n")
    info_out.write("outpath "+outpath+"\n")
    info_out.write("node_mindist "+str(node_mindist)+"\n")
    info_out.write("node_maxdist "+str(node_maxdist)+"\n")
    info_out.write("maxnbr "+str(maxnbr)+"\n")

info = {}
infofname = runpath+runname+".info"
stdout("Creating info dict", mpirank)
info = gt.get_info_dict(infofname)
if mpirank==0:
    for key,val in sorted(info.items()):
        stdout(str(key)+"\t"+str(val), mpirank)
        info_out.write(str(key)+" "+str(val)+"\n")

# Gather nodes
stdout("Generating nodes and edges...", mpirank)
tnode = time.time()
xmin,xmax = info["xmin"]/gt.long2km, info["xmax"]/gt.long2km
ymin,ymax = info["ymin"]/gt.lat2km, info["ymax"]/gt.lat2km
region_gsi = [xmin,xmax,ymin,ymax]

nodes, edges = gt.generate_nodes(region=region_gsi, mindist=node_mindist, 
    maxdist=node_maxdist, maxnbr=maxnbr, disable_tqdm=disable_tqdm)

tnode_ = time.time()
stdout("Done nodes and edges: "+str(tnode_ - tnode)+" seconds", mpirank)

n_nodes, n_edges = len(nodes.index), len(edges.index)
if mpirank == 0:
    stdout("Number of nodes "+ str(n_nodes), mpirank)
    info_out.write("n_nodes "+str(n_nodes)+"\n")
    stdout("Number of edges "+ str(n_edges), mpirank)
    info_out.write("n_edges "+str(n_edges)+"\n")

stdout("Generating velocity dataframe...", mpirank)
stdout("vel file contains "+str(info["n_points"])+" points", mpirank)

tvdf = time.time()
# Grab the hdf5 file and divide up
f5vel = h5py.File(runpath+runname+".hdf5", 'r', driver="mpio", comm=mpicomm)
# Check if they have the right amount of vel points
if f5vel.attrs["nvel"] != info["n_points"]:
    stdout("nvel from hdf5 ("+str(nvel)+") != nvel from info ("+str(info["n_points"])+")",mpirank)
    sys.exit(2)

# Use f5 nvel is not specified
if not nvel:
    nvel = f5vel.attrs["nvel"]

vdset = f5vel["veldat"]
if mpirank==0:
    info_out.write("nvel "+str(nvel)+"\n")

# Getting shape doesn't load to mem
#assert nvel == vdset.shape[0]
nvel_per = nvel//mpisize
vstart, vend = nvel_per*mpirank, nvel_per*(mpirank+1)
remainder = nvel%mpisize

# Divide up the dataset
if mpirank == (mpisize-1):
    vdata_np = vdset[vstart:vend+remainder]
else:
    vdata_np = vdset[vstart:vend]
stdout("With "+str(mpisize)+" processors, nvel_per = "+str(nvel_per)+\
    ", with remainder "+str(remainder),mpirank)

# I think we can close f5vel now
f5vel.close()
del vdset

# We can append all the vdfs after using comm.gather which creates a list
vdf = pd.DataFrame(columns=["day","tg","x_km","y_km","vx","vy","v","nodeID","dist2node","angle"])
gt.build_vdf(vdata_np,nodes,vdf,days=[],nTG=info["nTG"], nvel=nvel, disable_tqdm=disable_tqdm)
vdf.drop_duplicates(inplace=True)
mpicomm.Barrier()
vdflist = mpicomm.gather(vdf,root=0)
if mpirank==0:
    vdf = vdf.append(vdflist[1:], ignore_index=True)
    tvdf_ = time.time()
del vdflist
# Perhaps I should broadcast vdf?...what if it's very big...
print("Done: rank",mpirank)
vdf = mpicomm.bcast(vdf, root=0)

if mpirank==0:
    stdout(str(tvdf_ - tvdf)+" seconds", mpirank)
    stdout("Number of vel points "+str(len(vdf.index)), mpirank)
    info_out.write("Number of vel points"+str(len(vdf.index))+"\n")

nodes["ncar"] = 0
nodes["v_avg"] = 0.
nodes["v_std"] = 0.

edges["ncar_out"] = 0
edges["ncar_in"] = 0
edges["v_avg_out"] = 0.
edges["v_avg_in"] = 0.
edges["v_std_out"] = 0.
edges["v_std_in"] = 0.

if mpirank==0: 
    info_out.write("node feature header: [ncar, v_avg, v_std]"+"\n")
    info_out.write("edge feature header: [ncar_out, v_avg_out, v_std_out, ncar_in, v_avg_in, v_std_in]"+"\n")
    info_out.write("global feature header: [day, tg]"+"\n")

nsnap = 7*info["nTG"]

# Make output hdf5 file
h5out = h5py.File(outpath+runname+".hdf5", 'w', driver="mpio", comm=mpicomm) 
h5out.atomic = True
h5out.create_group("glbl_features")
h5out.create_group("node_features")
h5out.create_group("edge_features")
if mpirank==0:
    np_send = edges['sender'].values.copy().astype(np.int)
    np_rece = edges['receiver'].values.copy().astype(np.int)
    h5out.create_dataset("senders", data=np_send, dtype=np.int)
    h5out.create_dataset("receivers", data=np_rece, dtype=np.int)

if False:
    h5store = pd.HDFStore(outpath+runname+".hdf5", 'w', driver='mpio', comm=mpicomm)
    h5store.put("senders", edges["sender"].astype(np.int), format='fixed')
    h5store.put("receivers", edges["receiver"].astype(np.int), format='fixed')

tloop_start = time.time()
bcheck_velloop = False
bcheck_snaploop = False

stdout("Beginning feature construction loop", mpirank)
# Certain parts of the loop aren't great for parallelization
# e.g. nodes.at[idx,'ncar'] etc.
# There is substantial modding of the nodes and edges dicts here
# We should try to parallelize as many loops as possible
for day in trange(7, desc='Days     ', disable=disable_tqdm):
    for tg in trange(info["nTG"], desc='TimeGroup', disable=disable_tqdm):
        # Get velstats for this day, tg
        vdf_ = vdf[(vdf['day']==day) & (vdf['tg']==tg)]
        for idx, node in nodes.iterrows():
            # Give this subset of vels, calc stats for each node
            vels = vdf_[vdf_['nodeID'] == idx]
            nodes.at[idx,'ncar'] = len(vels.index)
            if len(vels.index) == 0:
                continue
            nodes.at[idx,'v_avg'] = vels.mean(axis=0)['v']
            if len(vels.index) > 1:
                nodes.at[idx,'v_std'] = vels.std(axis=0)['v']
        
            # Iterate over this nodes edges, adding vel stats as necessary
            edges_ = edges[edges["sender"] == idx]
            for eidx, e in edges_.iterrows():
                v_out, v_in = [], []
                for iv, v in vels.iterrows():
                    dtheta = v['angle'] - e['angle']
                    if (abs(dtheta) < 0.25*np.pi) | (abs(dtheta) > 1.75*np.pi):
                        v_out.append(v['v'])
                    if (abs(dtheta) > np.pi*0.75) & (abs(dtheta) < np.pi*1.25):
                        v_in.append(v['v'])
                
                if len(v_out) > 0:
                    edges.at[eidx, "ncar_out"] = len(v_out)
                    v_avg_out, v_std_out = np.mean(v_out), np.std(v_out)
                    edges.at[eidx, "v_avg_out"] = v_avg_out
                    edges.at[eidx, "v_std_out"] = v_std_out
                if len(v_in) > 0:
                    edges.at[eidx, "ncar_in"] = len(v_in)
                    v_avg_in, v_std_in = np.mean(v_in), np.std(v_in)
                    edges.at[eidx, "v_avg_in"] = v_avg_in
                    edges.at[eidx, "v_std_in"] = v_std_in
                    
        # Add to arrays
        isnap = (day*info["nTG"]) + tg
        if False:
            h5store.put("node_features/day"+str(day)+"tg"+str(tg),
                        nodes[["ncar","v_avg","v_std"]], 
                        format='fixed')
            h5store.put("edge_features/day"+str(day)+"tg"+str(tg),
                        edges[["ncar_out","v_avg_out","v_std_out","ncar_in","v_avg_in","v_std_in"]], 
                        format='fixed')
            h5store.put("glbl_features/day"+str(day)+"tg"+str(tg),
                        pd.DataFrame([[day,tg]],columns=["day","tg"],dtype=np.float), 
                        format='fixed')

        else:
            A = nodes[["ncar","v_avg","v_std"]].values.copy()
            B = edges[["ncar_out","v_avg_out","v_std_out","ncar_in","v_avg_in","v_std_in"]].values.copy()
            h5_A = h5out.create_dataset("node_features/day"+str(day)+"tg"+str(tg),
                        A.shape, dtype=np.float)
            h5_A = A[:]
                    
            h5out.create_dataset("edge_features/day"+str(day)+"tg"+str(tg),\
                        edges[["ncar_out","v_avg_out","v_std_out","ncar_in",\
                        "v_avg_in","v_std_in"]].values.copy())
            h5out.create_dataset("glbl_features/day"+str(day)+"tg"+str(tg),\
                        np.array([[day,tg]],dtype=np.int))


    if disable_tqdm:
        daytime = time.time()
        stdout("Day "+str(day)+" in "+str(daytime-tloop_start), mpirank)
        tloop_start = time.time()
        
mpicomm.Barrier()
if mpirank==0:
    h5store.close()

global_end = time.time()
if mpirank == 0:
    stdout("Total time for main loop: "+str(global_end - tloop_start)+" s", mpirank)
    info_out.write("Total time for main loop: "+str(global_end - tloop_start)+" s"+"\n")
    stdout("Total time for graphsnapper: "+str(time.time() - global_start)+" s", mpirank)
    info_out.write("Total time for graphsnapper: "+str(global_end - tloop_start)+" s"+"\n")
    info_out.close()


gt.MPI.Finalize()


