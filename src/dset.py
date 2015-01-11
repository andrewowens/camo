import numpy as np, img as ig, os, re, glob, mvg, copy

class Scan:
  # initialize from bundler
  def __init__(self, path, max_dim = 1000, frame_subset = None, use_cams_file = True):
  #def __init__(self, path, max_dim = None, frame_subset = None):
    self.path = os.path.abspath(path)
    self.bundle_file = os.path.join(self.path, 'bundle', 'bundle.out')
    self.view_prefix = 'view'
    # if os.path.exists(os.path.join(path, 'view1.jpg')):
    #   self.view_prefix = 'view'
    # else:
    #   assert os.path.exists(os.path.join(path, 'nocube_001.jpg'))
    #   self.view_prefix = 'nocube_'
      
    if not os.path.exists(self.path):
      raise RuntimeError('Path does not exist: %s' % self.path)

    self.im_files = sorted(glob.glob(os.path.join(self.path, '%s*.jpg' % self.view_prefix)), key = self.file_index)

    # check for missing files
    assert all(self.file_index(x) == 1+i for i, x in enumerate(self.im_files))

    if len(self.im_files) == 0:
      raise RuntimeError('No image files!')

    self.full_shape = ig.imread(self.im_files[0]).shape
    if max_dim is None:
      self.scale = 1.
    else:
      self.scale = min(float(max_dim+0.4) / np.array(self.full_shape[:2], 'd'))
    
    focals, Rt, tracks = read_bundler(self.bundle_file, self.full_shape)

    # convert to 1-based system (i.e. name it after the filename)
    ok_sfm = 1 + np.nonzero([f != 0 for f in focals])[0]
    
    ok_frames = [i for i in xrange(len(self.im_files)) if np.any(Rt[i][0] != 0)]

    # the case when a user explicitly marks image files that are OK
    if use_cams_file and os.path.exists(os.path.join(self.path, 'good_cams.txt')):
      marked_good = map(int, open(os.path.join(self.path, 'good_cams.txt'), 'r').read().split())
    else:
      marked_good = None

    ok_frames = set(range(1+len(self.im_files)))
    ok_frames = ok_frames.intersection(ok_sfm)
    if frame_subset is not None:
      ok_frames = ok_frames.intersection(frame_subset)
    if marked_good is not None:
      ok_frames = ok_frames.intersection(marked_good)

    ok_frames = sorted(ok_frames)

    # go back to 0-based scheme (i.e. from the one that matches the filenames)
    ok_frames = [x-1 for x in ok_frames]
    
    self.im_files = [self.im_files[f] for f in ok_frames]
    self.length = len(self.im_files)
    self.frames = range(self.length)
    
    if self.length == 0:
      raise RuntimeError('SfM failed: no good cameras!')
    
    self.Ks = np.zeros((self.length, 3, 3))
    self.Rs = np.zeros((self.length, 3, 3))
    self.ts = np.zeros((self.length, 3))
    for i, f in enumerate(ok_frames):
      # K = np.array([[self.scale*focals[f], 0., -self.scale*self.full_shape[1]/2.,],
      #               [0., -self.scale*focals[f], -self.scale*self.full_shape[0]/2.,],
      #               [0., 0., -1]])
      K = -np.array([[-self.scale*focals[f], 0., -0.5 + self.scale*self.full_shape[1]/2.,],
                     [0., self.scale*focals[f], -0.5 + self.scale*self.full_shape[0]/2.,],
                     [0., 0., 1.]])
      self.Ks[i] = K
      self.Rs[i], self.ts[i] = Rt[f]

    #self.im_cache = [None] * self.length
    self.im_cache = {}
    self.idx = map(self.file_index, self.im_files)

  
  def add_virtual_camera(self, K, R, t, im):
    # a hack
    self = copy.deepcopy(self)
    frame = len(self.Ks)
    self.length += 1
    self.Ks = np.concatenate([self.Ks, K[np.newaxis, :, :]], axis = 0)
    self.Rs = np.concatenate([self.Rs, R[np.newaxis, :, :]], axis = 0)
    self.ts = np.concatenate([self.ts, t[np.newaxis, :]], axis = 0)
    self.frames = range(self.length)
    self.im_cache[frame] = im
    
    return self, frame
                  
  def project(self, frame, X):
    return mvg.homog_transform(self.P(frame), X.T).T
    
  def im(self, frame):
    if frame not in self.im_cache:
      self.im_cache[frame] = ig.resize(np.array(ig.imread(self.im_files[frame]), 'd'), self.scale)
      
    return self.im_cache[frame].copy()
      
    #return ig.resize(np.array(ig.imread(self.im_files[frame]), 'd'), self.scale)
    #return ig.resize(ig.imread(self.im_files[frame]), self.scale)
    #return ig.resize(np.array(ig.imread(self.im_files[frame]), 'd'), (666, 1000))

  def K(self, frame): return self.Ks[frame]
  def R(self, frame): return self.Rs[frame]
  def P(self, frame):
    return np.dot(self.Ks[frame], np.hstack([self.Rs[frame], self.ts[frame, :, np.newaxis]]))

  def center(self, frame):
    return -np.dot(self.Rs[frame].T, self.ts[frame])

  # for when a dataset has images of a real proxy cube
  def has_real(self):
    return os.path.exists(self.real_file(0))
  
  def im_with_real(self, frame):
    if self.has_real():
      return ig.resize(ig.imread(self.real_file(frame)), self.scale)
    else:
      raise RuntimeError('Scan %snot contain a real cube' % self.path)

  def real_file(self, frame):
    base, fname = os.path.split(self.im_files[frame])
    return os.path.join(base, fname.replace('nocube', 'view'))

  def file_index(self, fname):
    return int(re.match(r'.*%s(\d+)\.jpg' % self.view_prefix, fname).group(1))

  def frames_from_idx(self, idx):
    return [self.idx.index(i) for i in idx]
    
def read_bundler(fname, im_shape):
  f = file(fname)
  f.readline()
  ncams = int(f.readline().split()[0])
  Rt = []
  focals = []
  for i in xrange(ncams):
    focals.append(float(f.readline().split()[0]))
    R = np.array([map(float, f.readline().split()) for x in xrange(3)])
    t = np.array(map(float, f.readline().split()))
    Rt.append((R, t))

  pt_tracks = []
  while True:
    line = f.readline()
    if line is None or len(line.rstrip()) == 0:
      break
    X = np.array(map(float, line.split()))
    f.readline() # color

    projs = f.readline().split()
    n = int(projs[0])
    track = []
    for j in xrange(n):
      frame = int(projs[0 + 1 + 4*j])
      x = im_shape[1]/2. + float(projs[2 + 1 + 4*j])
      y = im_shape[0]/2. - float(projs[3 + 1 + 4*j])
      track.append((frame, np.array([x, y], 'd')))
    pt_tracks.append((X, track))
    
  return focals, Rt, pt_tracks

def test_scan():
  dim = None
  scan = Scan('../data/couch', dim)
  # no missing frames
  assert scan.length == 5

  #assert scan.im(0).shape[1] == dim

  _, _, tracks = read_bundler(scan.bundle_file, scan.full_shape)
  import mvg
  for X, track in tracks:
    for frame, proj in track:
      diff = np.linalg.norm(scan.scale*proj - mvg.homog_transform(scan.P(frame), X))
      print proj, mvg.homog_transform(scan.P(frame), X), diff
      #assert diff <= 2.

def t_from_Rc(R, c):
  return -np.dot(R, c)

def c_from_Rt(R, t):
  return -np.dot(R.T, t)
