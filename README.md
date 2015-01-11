Code for the CVPR 2014 paper:
Camouflaging an Object from Many Viewpoints
Andrew Owens, Connelly Barnes, Alex Flint, Hanumant Singh, William Freeman.

- Requires some Python libraries (numpy, scipy, pylab, skimage,
  networkx), and Cython.  All of these come with the (free) Anaconda
  Python distribution: https://store.continuum.io/cshop/anaconda/.

- If you want to use the MRF-based models, you'll need to build the
  Cython code.  Just run "make", and it should compile (feel free to
  email us if there are any problems with this!). You'll also need to
  install GCO 3.0 (http://vision.csd.uwo.ca/code/gco-v3.0.zip).  You
  can set it up by running "lib/download.sh"

- To try out the code, run "camo.test_camo()".  This camouflages a box
  and generates a webpage that shows each viewpoint.  

- Please let us know if you have any questions about the code!