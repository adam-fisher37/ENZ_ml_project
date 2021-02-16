# run this .py file to gen the training data
# just here so you dont have to put the like 2 lines of code in a bunch

# boilerplate
import gendata_char as gen
import numpy as np

# variables
ang = np.array([55.,60,65])
com = 'angles: 55-5-65deg, dataset = data, wvl = [400,1000]nm'
setlen = 500

# run the boi
gen.gen_dat(ang,setlen,com,'data',core=False)