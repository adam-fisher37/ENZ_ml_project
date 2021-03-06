# Hello and welcome to ENZ machine learning characterization project!

Est. 2021

This was made for use by Professor Giuseppe Strangi's NANOPLASM Lab at Case Western Reserve University (CWRU). Note, this repo was made by Adam Fisher and its views do not reflect that of CWRU or the NANOPLASM Lab.

Mission Statement: to use DNN for inverse design purposes to characterize multistack metamaterials with intended epsilon-near-zero (ENZ) resonance.

Current ENZ design is 5x{Al2O3,Ag,Ge} on a glass substrate made at the CWRU MORE Center using electron beam and thermal evaporation deposition techniques. 

BRANCH SYSTEM: main is for CODE ONLY, create a new branch with the pc name or last name and keep any data files on that branch, doing this to keep the main branch clean and to prevent smaller pc's with less storage getting messed up from massive .h5 files

NOTE: the index of refraction for these materials can either be generated with theoretical models using: 'BB_metals.py', 'LD_metals.py', 'dielectric_materials.py'. OR can use experimentally derived tabulated data for the three ENZ materials in the folder 'materials', these were modeled by NANOPLASM Lab and Woolam to reflect the indices of refraction of the materials deposited in the MORE Center, with the most accurate being the ones with 'case_...' file names. 

NOTE: index of refraction convention with an imaginary component is, n = n + ik, with i as the imaginary unit.