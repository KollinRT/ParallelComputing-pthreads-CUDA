# How to run
Can run all the scripts by running the `./runAll.sh` script which will cycle through all of them. At the end, will have several output files outputted into each respected build folder. Copy paste all of the data row entries into one of them and then can run and analyze the data with the scripts in Python. Data is already collected and pooled together in the `Python/data/results.csv` file. 

If wanting to run individual code sections, make sure comment out the continuation to the one at the bottom of the respective `./run.sh` files, so it does not launch the next in line. Order is `Lab1CPP` -> `Lab2CPP-CUDA` -> `Lab2CPP-pthreads-Elements` -> `Python`.

# How to analyze
Navigate to the Python folder and run the  `analyzeData.py` and make sure that pandas, matplotlib, and seaborn are installed. Make sure the the function call in ` df = pd.read_csv("./data/results.csv")` is pointing to the right data collection point.

POSSIBLY MAKE SURE THAT `data` and `figures` directories are created already.

# Erorrs
If permission is not allowed with the `./run{py}.sh` scripts, then run `rm -rf !(run.sh)` or remove all except `run.sh` file for the `CUDA/C++` files, and then make sure `cmake ..` and `make` are ran in order to regenerate the `CMakeCache` files in the build section then double check that the respective `*.sh` scripts have had `chmod +x run.sh` to enable permissions.
