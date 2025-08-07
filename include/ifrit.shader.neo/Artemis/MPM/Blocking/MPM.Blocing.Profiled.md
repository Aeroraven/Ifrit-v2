114514 particles @ gtx1050,8 iters,3 substeps,64^3 grid
global atomics:                     82.02ms
block + shared atomics (fp32 cas):  56.52ms 
block + shared atomics (fp32):      50.52ms (-38.4%)
block + shared atomics (int32):     33.35ms (-59.3%)

50000 particles @ gtx1050,8 iters,3 substeps,64^3 grid
global atomics:                     37.55ms
block + shared atomics (fp32 cas):  27.49ms 
block + shared atomics (fp32):      24.52ms (-34.7%)
block + shared atomics (int32):     16.42ms (-56.3%)

20ms @ rtx3070,8 iters,3 substeps,64^3 grid
global atomics:                     130432 particles
block + shared atomics (fp32):      166272 particles (+27.4%)
block + shared atomics (int32):     210560 particles (+61.4%)


114514 particles @ gtx1050,8 iters,3 substeps,64^3 grid
block + shared atomics (fp32 cas):  56.52ms 
block + shared atomics (fp32):      50.52ms

50000 particles @ gtx1050,8 iters,3 substeps,64^3 grid
block + shared atomics (fp32 cas):  27.49ms 
block + shared atomics (fp32):      24.52ms