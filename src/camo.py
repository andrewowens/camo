import dset, img as ig, box, os, util as ut, numpy as np, networkx as nx, scipy.ndimage, glob, skimage.color
na = np.newaxis

try:
  # mrf is a Cython module that depends on the gco graph cuts library
  import mrf
except ImportError:
  print 'mrf import failed! Cannot use the MRF-based models.  Try compiling it with "make". '

METHODS = 'random greedy interior boundary mean uniform'.split()
  
def camo(scan, mesh, method, **kwargs):
  if method == 'greedy':
    return greedy_project(scan, mesh)
  elif method == 'random':
    return random_project(scan, mesh)
  elif method == 'order':
    return random_project(scan, mesh, **kwargs)
  elif method == 'mean':
    return mean_project(scan, mesh)
  elif method == 'uniform':
    return uniform_project(scan, mesh)
  elif method == 'interior':
    return interior_mrf(scan, mesh)
  elif method == 'boundary':
    return boundary_mrf(scan, mesh)

def to_color_space(x):
  return skimage.color.rgb2lab(x)

def from_color_space_2d(x):
  return np.uint8(255*np.squeeze(skimage.color.lab2rgb(np.array(x[np.newaxis, :, :], 'float64'))))

def to_color_space_2d(x):
  return np.squeeze(skimage.color.rgb2lab(np.array(x[np.newaxis, :, :])))

color_type = 'float32'

MAX_STABLE = 1e4

def uniform_project(scan, mesh):
  vis = np.zeros(mesh.ntexels, 'bool')
  colors = []
  for frame in xrange(scan.length):
    visible = mesh.texel_visible(scan, frame)
    projs = scan.project(frame, mesh.texel_pts)
    colors += list(ig.lookup_bilinear(scan.im(frame), projs[visible, 0], projs[visible, 1]))
    vis = np.logical_or(vis, visible)
    
  texel_colors = np.zeros((mesh.ntexels, 3))
  texel_colors[vis] = np.mean(colors, axis = 0)
  
  return [texel_colors]

def mean_project(scan, mesh):
  counts = np.zeros(mesh.ntexels)
  texel_colors = np.zeros((mesh.ntexels, 3))
  for frame in xrange(scan.length):
    visible = mesh.texel_visible(scan, frame)
    projs = scan.project(frame, mesh.texel_pts)
    counts += visible
    texel_colors[visible] += ig.lookup_bilinear(scan.im(frame), projs[visible, 0], projs[visible, 1])
  texel_colors[counts > 0] /= np.array(counts[counts > 0][:, np.newaxis], 'd')
  return [texel_colors]

def random_project(scan, mesh, order = None):
  # totally random projection order
  if order is None:
    order = ut.shuffled(range(scan.length))
  filled = np.zeros(mesh.ntexels, 'bool')
  fill = np.nan # 0
  texel_colors = fill + np.zeros((mesh.ntexels, 3))
  print 'projection order =', order
  for frame in order:
    ok = mesh.texel_visible(scan, frame)
    ok[filled] = 0
    projs = scan.project(frame, mesh.texel_pts)
    if np.isnan(fill):
      assert(np.all(np.isnan(texel_colors[ok])))
    else:
      assert(np.all(texel_colors[ok] == fill))
    texel_colors[ok] = ig.lookup_bilinear(scan.im(frame), projs[ok, 0], projs[ok, 1])
    filled[ok] = 1
  return [texel_colors]

def greedy_project(scan, mesh, stable_angle = np.radians(70)):
  filled = np.zeros(mesh.ntexels, 'bool')
  texel_colors = np.zeros((mesh.ntexels, 3))

  visible_by_view = np.zeros((scan.length, mesh.ntexels), 'bool')
  stable_by_view = np.zeros((scan.length, mesh.ntexels), 'bool')
  
  for frame in xrange(scan.length):
    visible = mesh.texel_visible(scan, frame)
    visible_by_view[frame] = visible
    
    for j in xrange(mesh.nfaces):
      d = np.dot(ut.normalized(-mesh.face_center[j] + scan.center(frame)), mesh.face_planes[j][:3])
      assert -1 <= d <= 1
      angle = np.arccos(abs(d))
      print frame, j, np.rad2deg(angle), (angle <= stable_angle)
      on_face = (mesh.tex2juv[:, 0] == j)
      stable_by_view[frame, on_face] = visible_by_view[frame, on_face]*(angle <= stable_angle)
    print 'stable texels', np.sum(stable_by_view[frame])

  projectable = stable_by_view.copy()
  unstable = np.all(-stable_by_view, axis = 0)
  print 'Unstable faces:', np.unique(mesh.tex2juv[unstable, 0])
  projectable[:, unstable] = visible_by_view[:, unstable]

  unused = range(scan.length)
  while np.sum(projectable) > 0:
    # it's ok to project onto a face if it's (1) unfilled (2) it's either stable, or it's unstable in every view
    frame = max(ut.shuffled(unused), key = lambda f : np.sum(projectable[f, :]))

    projs = scan.project(frame, mesh.texel_pts)
    ok = projectable[frame]
    print 'chose', frame, np.sum(projectable[frame, :]), np.sum(ok), 'projecting to', np.unique(mesh.tex2juv[projectable[frame, :], 0])
    texel_colors[ok] = ig.lookup_bilinear(scan.im(frame), projs[ok, 0], projs[ok, 1])
    projectable[:, ok] = 0

    unused.remove(frame)
    print 'projected', frame, np.sum(filled)
      
  return [texel_colors]

def make_graph(mesh):
  u1, v1 = np.mgrid[:mesh.texsize, :mesh.texsize]
  u1, v1 = u1.flatten(), v1.flatten()
  adj = []

  dus = [0, 1]
  dvs = [1, 0]
  for j in xrange(mesh.nfaces):
    for du, dv in zip(dus, dvs):
      u2 = u1 + du
      v2 = v1 + dv
      ok = ut.index_in_bounds(mesh.juv2tex.shape[1:], u2, v2)
      inds1 = mesh.juv2tex[j, u1[ok], v1[ok]]
      inds2 = mesh.juv2tex[j, u2[ok], v2[ok]]
      adj += zip(inds1, inds2)
  
  for j in xrange(mesh.nfaces):
    face_border = np.logical_and(mesh.on_border, mesh.tex2juv[:, 0] == j)
    other_border = np.logical_and(mesh.on_border, mesh.tex2juv[:, 0] != j)

    dist, knn_inds = ut.knnsearch(mesh.texel_pts[other_border], mesh.texel_pts[face_border], k = 3)
    inds1 = np.nonzero(face_border)[0]
    inds2 = np.nonzero(other_border)[0]
    i, j = np.nonzero(dist <= 0.000001)
    adj += zip(inds1[i], inds2[knn_inds[i, j]])


  graph = nx.Graph()
  graph.add_nodes_from(range(mesh.ntexels))
  graph.add_edges_from(adj)

  if 0:
    assert 2*mesh.ntexels == graph.number_of_edges()
  
  return graph

def make_labels(scan, shift_dim, shift_dist):
  labels = []
  for frame in xrange(scan.length):
    for x in xrange(-shift_dim, shift_dim+1):
      for y in xrange(-shift_dim, shift_dim+1):
        labels.append((frame, shift_dist*x, shift_dist*y))
  return np.array(labels, 'l')

def outline_mask(scan, mesh, frame, thresh):
  mask = box.mask(scan, mesh, frame)
  D = scipy.ndimage.distance_transform_edt(1-mask)
  return np.logical_and(1 <= D, D <= thresh)
  
def occlusion_mask(scan, mesh, frame, thresh = 2., outline = False):
  mask = box.mask(scan, mesh, frame)
  D = scipy.ndimage.distance_transform_edt(mask)
  return D <= thresh
  
def project_texels(scan, frame, mesh, im, geom, shift = np.zeros(2), order = 1, mode = 'reflect', cval = 0.0, invisible_colors = False):
  if invisible_colors:
    vis = np.ones(mesh.ntexels, 'bool')
  else:
    vis = geom.texel_visible(frame)
    
  shift = np.asarray(shift)
  proj = shift + scan.project(frame, mesh.texel_pts[vis])
  colors = ig.lookup_bilinear(im, proj[:, 0], proj[:, 1], order = order, mode = mode, cval = cval)

  all_colors = np.zeros((mesh.ntexels,) + colors.shape[1:], dtype = colors.dtype)
  all_colors[vis] = colors
  
  return vis, all_colors

def label_colors(scan, mesh, geom, labels, sigma = 0, invisible_colors = False):
  colors = np.zeros((mesh.ntexels, len(labels), 3), color_type)

  ims = {}
  for frame in np.unique(labels[:, 0]):
    if sigma == 0:
      ims[frame] = to_color_space(scan.im(frame))
    else:
      ims[frame] = ig.blur(to_color_space(scan.im(frame)), sigma)
      
  for p, (frame, dx, dy) in enumerate(labels):
    vis, c = project_texels(scan, frame, mesh, ims[frame], geom, (dx, dy), invisible_colors = invisible_colors)
    colors[vis, p] = c[vis]

  label_valid = np.zeros((mesh.ntexels, len(labels)), 'bool')
  
  # for a given texel, a label is valid if the corresponding frame sees the texel
  for p, (frame, _, _) in enumerate(labels):
    vis = geom.texel_visible(frame)
    label_valid[vis, p] = True
    
  return colors, label_valid
  
def occlusion_texels(scan, mesh, frame, thresh = 1., only_border = True):
  occ_mask = np.array(occlusion_mask(scan, mesh, frame, thresh = thresh), 'd')
  vis = mesh.texel_visible(scan, frame)
  proj = np.array(scan.project(frame, mesh.texel_pts), 'l')
  occ = np.zeros(mesh.ntexels, 'bool')
  occ[vis] = occ_mask[proj[vis, 1], proj[vis, 0]]

  if only_border:
    occ = ut.land(occ, mesh.on_border)

  assert np.all(vis[occ])

  return occ

def occlusion_costs(scan, mesh, labels, geom, sigma = 4., weight_by_frame = True, full_occ = False):
  occ_border = int(0.1*mesh.texsize)
  label_color, label_valid = label_colors(scan, mesh, geom, labels, sigma = sigma, invisible_colors = False)

  assert weight_by_frame
  
  occ_samples = np.zeros((mesh.ntexels, scan.length, 3))
  has_sample = np.zeros((mesh.ntexels, scan.length), 'bool')
  
  for frame in scan.frames:
    if full_occ: 
      frame_occ = geom.texel_visible(frame)
    else:
      frame_occ = occlusion_texels(scan, mesh, frame)

      if occ_border is not None:
        as_juv = mesh.index_as_juv(frame_occ).copy()

        for j in xrange(as_juv.shape[0]):
          dist, ind = scipy.ndimage.distance_transform_edt(1 - as_juv[j], return_indices = True)
          dist[ind[0] < 0] = np.inf
          as_juv[j, dist <= occ_border] = True

        frame_occ = np.logical_and(geom.texel_visible(frame), mesh.index_as_flat(as_juv))
      
    vis, colors = project_texels(scan, frame, mesh, ig.blur(to_color_space(scan.im(frame)), sigma), geom)
    
    assert np.all(vis[frame_occ])
    
    occ_samples[frame_occ, frame, :] = colors[frame_occ]
    has_sample[frame_occ, frame] = True

  is_occ = np.any(has_sample, axis = 1)

  costs = np.zeros((mesh.ntexels, label_valid.shape[1]), 'd')

  frame_nobs = np.array(np.sum(has_sample, axis = 0), 'd')
  nvisible = np.sum(np.any([geom.texel_visible(frame) for frame in scan.frames], axis = 0))
  assert np.all(frame_nobs > 0)
  frame_weights = float(nvisible)/scan.length*(1./frame_nobs)
  
  for frame in scan.frames:
    frame_samples = occ_samples[:, frame]

    dist_from_sample = np.sqrt(np.sum((label_color[is_occ, :, :] - frame_samples[is_occ][:, na, :])**2, axis = -1))
    dist_from_sample[-label_valid[is_occ]] = 0
    dist_from_sample[-has_sample[is_occ, frame]] = 0
    
    costs[is_occ, :] += frame_weights[frame]*dist_from_sample
    
  assert np.all(costs[-label_valid] == 0)

  costs = np.array(costs, 'float32')
  return costs

def face_visibility(scan, mesh):
  return np.array([box.faces_visible(scan, mesh, frame) for frame in scan.frames])

def stability_costs(scan, mesh, labels, scale = 10., thresh = 2., max_stable = MAX_STABLE):
  J_intra_view = np.zeros((scan.length, scan.length, mesh.nfaces, 2, 2))
  face_vis = face_visibility(scan, mesh)
  
  for frame1 in scan.frames:
    for frame2 in scan.frames:
      for j in xrange(mesh.nfaces):
        if not face_vis[frame1, j] or not face_vis[frame2, j]:
          J_intra_view[frame1, frame2, j] = np.eye(2)
        else:
          J_i = np.eye(2)
          J_ip = np.eye(2)
          for d in (0, 1):
            delta = np.array([1., 0]) if d == 0 else np.array([0., 1.])
            center = np.array([mesh.texsize/2., mesh.texsize/2.])
            uv1 = center - delta
            uv2 = center + delta
            h = abs(uv2[0] - uv1[0]) + abs(uv2[1] - uv1[1])

            pt1 = mesh.texel_pts[mesh.juv2tex[j, uv1[0], uv1[1]]]
            pt2 = mesh.texel_pts[mesh.juv2tex[j, uv2[0], uv2[1]]]
            f1_i = scan.project(frame1, pt1)
            f2_i = scan.project(frame1, pt2)
            f1_ip = scan.project(frame2, pt1)
            f2_ip = scan.project(frame2, pt2)

            J_i[:, d] = (f2_i - f1_i)/h
            J_ip[:, d] = (f2_ip - f1_ip)/h

          # use Jacobian chain rule to eliminate [du dv] variables
          J_intra_view[frame1, frame2, j] = np.dot(J_ip, np.linalg.pinv(J_i))

  # penalty for assigning a texel on face j a color from frame1
  stretch_penalty = np.zeros((scan.length, mesh.nfaces))
  for frame1 in scan.frames: #label
    for j in xrange(mesh.nfaces): # face

      costs = []
      for frame2 in scan.frames:
        if face_vis[frame1, j] and face_vis[frame2, j]:
          J = J_intra_view[frame1, frame2, j]
          eigs = np.abs(np.linalg.svd(J)[1])
          costs.append(np.minimum(30., scale*np.sum(np.maximum(eigs - thresh, 0)**2)))
      costs = np.array(costs, 'd')

      print frame1, j, costs
      
      stretch_penalty[frame1, j] = min(np.sum(costs), max_stable)/float(scan.length)

  return stretch_penalty[labels[:, 0][na, :], mesh.tex2juv[:, 0][:, na]]

def face_weights(scan, mesh, labels):
  face_vis = np.array([box.faces_visible(scan, mesh, frame) for frame in scan.frames], 'd')
  print face_vis, np.sum(face_vis, axis = 0), np.sum(face_vis), mesh.nfaces
  counts = np.sum(face_vis, axis = 0)/np.sum(face_vis)*mesh.nfaces
  return counts

class MRF:
  pass

def solve_face_mrf(mesh, m):
  # collapse each face into a superpixel
  face_data = np.zeros((mesh.nfaces, m.nlabels), 'float32')
  for j in xrange(mesh.nfaces):
    face_data[j] = np.sum(m.data_cost[mesh.tex2juv[:, 0] == j], axis = 0)

  mesh.tex2juv[m.edges[:, 0]]
    
  assert m.smooth_prior == 0
    
def boundary_mrf(scan, mesh, shift_dim = 0, shift_dist = 1, max_stable = MAX_STABLE,
                  occ_weight = 0.5, stable_weight = 1.,
                  per_term_stable = np.inf, use_face_mrf = True):
  """ Run the Boundary MRF model """
  geom = Geom(scan, mesh)
  labels = make_labels(scan, shift_dim, shift_dist)
  label_color, label_valid = label_colors(scan, mesh, geom, labels)
  
  stable_cost = stable_weight*stability_costs(scan, mesh, labels)

  if occ_weight == 0:
    print 'no occ cost!'
    occ_cost = np.zeros((mesh.ntexels, len(labels)))
  else:
    occ_cost = occ_weight*occlusion_costs(scan, mesh, labels, geom)

  node_visible = np.any(label_valid, axis = 1)

  data_cost = np.array(occ_cost + stable_cost, 'float32')
  
  data_cost[-label_valid] = 1e5
  # no valid label? any label will do
  data_cost[-node_visible] = 0

  assert np.max(data_cost[label_valid]) <= 1e3

  mismatch_cost = 1e5
  sameview_prior = 1e7 + np.zeros(mesh.ntexels, 'float32')
  sameview_prior[mesh.on_border] = mismatch_cost

  m = MRF()
  m.edges = make_graph(mesh).edges()
  m.data_cost = data_cost
  m.labels = labels
  m.label_color = label_color
  m.node_visible = node_visible
  m.sameview_prior = sameview_prior
  m.smooth_prior = 0.
  
  def en(r):
    u = np.sum(np.array(data_cost, 'd')[range(len(r)), r])
    s = np.sum(mismatched(m, r)*mismatch_cost)
    # 2* is to deal w/ bug in gc energy estimate
    print u + 2*s, (u, 2*s)
  
  if use_face_mrf:
    # Use a brute-force solver (written in Cython)
    results = mrf.solve_face_mrf(mesh, m)
    print 'Energy for brute-force solver', en(results)
    assert len(np.unique(results[m.node_visible])) <= 2
    if 0:
      # verify that the brute-force solver gets a better result than alpha-expansion
      results2 = mrf.solve_mrf(m)
      print 'Energy for alpha-expansion solver:', en(results2)
      assert en(results) <= en(results2)
  else:
    # Solve using alpha-expansion (requires gco)
    results = mrf.solve_mrf(m)

  colors = from_color_space_2d(label_color[range(len(results)), results])
  ut.toplevel_locals()

  occ_total = np.sum(occ_cost[range(len(results)), results])
  stable_total = np.sum(stable_cost[range(len(results)), results])
  print 'Occlusion cost:', occ_total
  print 'Stability cost:', stable_total
  
  return colors, results, labels, (occ_total, stable_total)

def mismatched(mrf, results):
  e = np.array(mrf.edges)
  ok_edges = mrf.node_visible[e[:, 0]]*mrf.node_visible[e[:, 1]]
  return results[e[ok_edges, 0]] != results[e[ok_edges, 1]]
  
def analyze_costs(scan, mesh, results, costs, texel_colors, label_info, cost_names = None, smooth_faces = True, frames = None):
  """ Visualize which costs contributed to the solution (warning: slow) """
  
  if frames is None:
    frames = range(scan.length)#sorted(set(map(int, np.linspace(0, scan.length-1, 10))))
    
  if cost_names is None:
    cost_names = range(len(costs))

  subcosts = np.array([cost[range(len(results)), results].copy() for cost in costs])
  #total = float(np.max(subcosts))
  
  print 'costs:'
  table = [cost_names, map(ut.pr, [np.sum(x) for x in subcosts])]
  
  if smooth_faces:
    for subcost in subcosts:
      print 'subcost', np.sum(subcost)
      as_juv = np.squeeze(mesh.index_as_juv(subcost))
      for j in xrange(as_juv.shape[0]):
        D, I = scipy.ndimage.distance_transform_edt(as_juv[j] == 0, return_indices = True)
        res = np.where(D < 20, as_juv[j][I[0], I[1]], as_juv[j]).copy()
        as_juv[j, :, :] = res
      print np.max(as_juv)
      subcost[:] = mesh.index_as_flat(as_juv).flatten()

  labeling_colors = np.array(ut.distinct_colors(len(label_info)))
  labeling_colors[ut.land(label_info[:, 1] == 0, label_info[:, 2] == 0)] = 255

  for frame in frames:
    row = [mesh.render(scan, frame, texel_colors)]

    frame_lc = labeling_colors.copy()
    frame_lc[label_info[:, 0] != frame] = 0
    row.append(mesh.render(scan, frame, frame_lc[results]))

    for subcost in subcosts:
      total = float(np.max(subcost))
      colors = np.clip(255*np.tile(subcost[:, np.newaxis], (1, 3))/total, 0, 255)
      row.append(mesh.render(scan, frame, colors))
      
    table.append(row)
  ig.show(table)

class Geom:
  """ A data structure that's used for caching visibility information """
  def __init__(self, scan, mesh, n = None):
    self.visible = np.array([mesh.texel_visible(scan, frame) for frame in scan.frames])
    
  def texel_visible(self, frame):
    return self.visible[frame]

def interior_mrf(scan, mesh, shift_dim = 1, shift_dist = 30.,
                 data_weight = 1., full_occ = 0,
                 occ_border = None, occ_weight = 0.5):
  """ Runs the Interior MRF camouflage method """
  # controls the importance of smoothness vs. the local evidence
  data_weight = 1./4*(64./mesh.texsize)
  
  geom = Geom(scan, mesh)
  labels = make_labels(scan, shift_dim, shift_dist)
  label_color, label_valid = label_colors(scan, mesh, geom, labels, invisible_colors = True)
  stable_cost = stability_costs(scan, mesh, labels)

  occ_cost = occ_weight*occlusion_costs(scan, mesh, labels, geom)
  
  face_weight = np.ones(mesh.nfaces) 
  node_visible = np.any(label_valid, axis = 1)
  fw = face_weight[mesh.tex2juv[:, 0]][:, na]
  data_cost = np.array(fw*(occ_cost + stable_cost), 'float32')
  data_cost[-label_valid] = 1e5
  # no invalid label? any label will do
  data_cost[-node_visible] = 0
  data_cost *= data_weight

  m = MRF()
  m.edges = make_graph(mesh).edges()
  m.data_cost = data_cost
  m.labels = labels
  m.label_color = label_color
  m.node_visible = node_visible
  m.sameview_prior = np.zeros(mesh.ntexels, 'float32')
  m.smooth_prior = 1.

  results = mrf.solve_mrf(m)

  assert np.all(np.logical_or(-node_visible, label_valid[range(len(results)), results]))
  colors = from_color_space_2d(label_color[range(len(results)), results])
  ut.toplevel_locals()

  occ_total = np.sum((fw*occ_cost)[range(len(results)), results])
  stable_total = np.sum((fw*stable_cost)[range(len(results)), results])
  print 'Occlusion cost:', occ_total
  print 'Stability cost:', stable_total
  
  return colors, results, labels, (occ_total, stable_total)

def test_camo(scene = '../data/charlottesville-3', method = 'random'):
  """ Camouflage a box in a given scene using the specified method.  Generates a webpage showing the result """
  scan = dset.Scan(scene)
  mesh = box.load_from_mat(os.path.join(scene, 'cube.mat'))
  texel_colors = camo(scan, mesh, method)[0]
  box.render_scene(scene, texel_colors, mesh)

def test_all(scene = '../data/bookshelf-real'):
  for method in METHODS:
    print 'Method:', method
    test_camo(scene, method)
