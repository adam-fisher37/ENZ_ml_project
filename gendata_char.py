# generation of training data for the ENZ characterization ML project
# by Adam Fisher, started 1/25/21
# NOTE: make sure you have all the packages nessecary to run and to do ML (keras, tensorflow)
# ^ all .txt and .py files are already part of this repo

# boilerplate
import numpy as np
import h5py
import datetime
import multiprocessing
from joblib import Parallel, delayed
import TMM as tmm
import LD_metals as ld
import dielectric_materials as di
from numba import jit

# import mats
# wvl starts as nm, make m
(wvl,nag,kag) = np.loadtxt('./materials/case_genosc.txt',usecols=(0,1,2),unpack=True)
(nal2o3,kal2o3) = np.loadtxt('./materials/case_al2o3.txt',usecols=(1,2),unpack=True)
(nge,kge) = np.loadtxt('./materials/case_ge.txt',usecols=(1,2),unpack=True)
(ngl,kgl) = np.loadtxt('./materials/case_gl.txt',usecols=(1,2),unpack=True)
wvl = wvl*1.0e-9
# using index of refraction n + ik
n_ag = np.array((nag+1j*kag),dtype=complex)
n_al2o3 = np.array((nal2o3+1j*kal2o3),dtype=complex)
n_ge = np.array((nge+1j*kge),dtype=complex)
n_gl = np.array((ngl+1j*kgl),dtype=complex)
# put mats in an array
materials = np.array([n_al2o3,n_ag,n_ge])
# set up rng, using np.random's new generator API
rng = np.random.default_rng()
sig = .01
# range you are sampling over
al_range = np.array([1.,70.1])*1e-9
ag_range = np.array([1.,30.1])*1e-9
ge_range = np.array([.1,3])*1e-9

# important variables
l = np.ones(15)
num_mat = len(materials)


def gen_dat(ang, set_length, comments, dset, core=True):
	'''
	NOTE: currently this program should only be run once a day!
	This generates and  saves (as an .hdf5 file) the training data for a 5x tri-layer nonuniform metamaterial
	calculates rp, rs, tp, ts, psi, delta over user-defined angles
	pretty much fire and forget so has a bunch of assumptions
	assumptions: structure: 
	5x{Al2O3, Ag, Ge} (top->bottom)
	substrate: glass, superstrate: air
	thicknesses are randomly choosen over uniform distribution
	when simulating exp data, adds gaussian errors of .01 for all data points
	assuming this can handle multiprocessing and has multiple cores
	INPUTS:
	ang - array, [A] - the angle (deg) of incidence, use what you would use on your ellipsometer
	set_length - integer - number of structures you are simulating, recomend .5 mil
	comments - string - any extra comments you would like to add to the .txt file that gives details about the data generated
	^NOTE: running this multiple times in 1 day will not cause an overwrite because there is an append feature when writing to .txt files
	dset - string - the name of the dataset that this batch of results will be placed in
	^NOTE: if you run htis program multiple times in 1 day without changing this input for each run, it will overwrite any previous data, as I do not believe that h5py has an append capability
	core - bool - defualt True, determines whether to use multiple cores or just 1 (for testing/debuging)
	OUTPUTS:
	none - will generate and save the data and then close the file in this funct
	'''
	@jit(nopython=True)
	def gen():
		'this is the function that generates the data and will be iterated over'
		# set up variables and allocate storage
		n = np.zeros((l.size,wvl.size),dtype=complex)
		psi = delta = np.zeros((ang.size*wvl.size))
		ellips = np.zeros((2,ang.size*wvl.size))
		rp = rs = tp = ts = psi = np.zeros_like(psi)
		# creating structure, mats are not random, only thicknesses
		n[::3] = materials[0]
		n[1::3] = materials[1]
		n[2::3] = materials[2]
		l[::3] = rng.uniform(low=al_range[0],high=al_range[1],size=l[::3].size)
		l[1::3] = rng.uniform(low=ag_range[0],high=ag_range[1],size=l[1::3].size)
		l[2::3] = rng.uniform(low=ge_range[0],high=ge_range[1],size=l[2::3].size)
		# calculate sim exp data
		# p-pol
		rp = np.array([tmm.reflect_amp(1,ang[j],wvl[i],n[:,i],l,1.,n_gl[i]) for j in range(len(ang)) for i in range(len(wvl))]) + rng.normal(scale=sig,size=rp.size)
		tp = np.array([tmm.trans_amp(1,ang[j],wvl[i],n[:,i],l,1.,n_gl[i]) for j in range(len(ang)) for i in range(len(wvl))]) + rng.normal(scale=sig,size=tp.size)
		# s-pol
		rs = np.array([tmm.reflect_amp(0,ang[j],wvl[i],n[:,i],l,1.,n_gl[i]) for j in range(len(ang)) for i in range(len(wvl))]) + rng.normal(scale=sig,size=rs.size)
		ts = np.array([tmm.trans_amp(0,ang[j],wvl[i],n[:,i],l,1.,n_gl[i]) for j in range(len(ang)) for i in range(len(wvl))]) + rng.normal(scale=sig,size=ts.size)
		# ellips
		ellips = np.array([tmm.ellips(ang[j],wvl[i],n[:,i],l,1.,n_gl[i]) for j in range(len(ang)) for i in range(len(wvl))])
		# add gaussian errors and split into psi and delta
		psi = ellips[:,0] + rng.normal(scale=sig,size=psi.size)
		delta = ellips[:,1] + rng.normal(scale=sig,size=delta.size)
		# save data as a long straight line
		data = np.reshape(np.concatenate((ang,l,rp,rs,tp,ts,psi,delta)),(1,ang.size + l.size + 6*ang.size*wvl.size))
		return data

	# make file
	date = datetime.datetime.now()
	# file name, for both the comments and h5 file, maybe want to have first lil string be an input
	fname = 'traindata_enzchar'+'_lay'+str(l.size)+'_mats'+str(num_mat)+'_n'+str(set_length/1000)+'k_'+date.strftime("%Y")+date.strftime("%m")+date.strftime("%d")
	# leave some comments so you know wtf you were doing here
	com = '# Materials: Al2O3, Ag, Ge. nonuniform trange: [1,70],[1,30],[.1,3]nm respectively. return [ang,l,Rp,Rs,Tp,Ts,ellipsometric]\n'
	com1 = '# datasets: 1\n'
	# ^ names of the datasets bc may use more than 1 per file
	with open(fname+'comments.txt','a') as f:
		f.write(com)
		f.write(com1)
		f.write(comments+'\n')
	# count cores for multiprocessing
	if core:
		cores = 6
		# cores = multiprocessing.cpu_count() # currently hard-set will change later
	else:
		cores = 1
	print('Will parallelize with %d cores.\n'%(cores))
	print('doing that data thang, please wait...')
	# if its more useful when recalling data may want to make a loop so it creates multiple dataset per file
	res = Parallel(n_jobs=cores)(delayed(gen)() for i in range(set_length))
	print('puttin that mf in ya file')
	# create .hdf5 file
	with h5py.File(fname+'.hdf5','w') as h:
		dat = h.create_dataset(dset,data=res)
	del res
	print('congrats u done!')
	return