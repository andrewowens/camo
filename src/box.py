TEXSIZE = 256

import numpy as np, util as ut, mvg, scipy.io, dset, os, img as ig

na = np.newaxis

ALIAS_SAMPLES = 5

class Mesh:
  def __init__(self, face_idx, mesh_pts, texsize = TEXSIZE):
    if texsize is None:
      texsize = 256
      
    self.face_idx = np.asarray(face_idx)
    self.nfaces = len(face_idx)
    self.mesh_pts = np.asarray(mesh_pts)
    
    self.texsize = texsize
    self.face_planes, self.face_center, self.face_edges, self.face_pts, self.face_juv = [], [], [], [], []
    
    u_grid, v_grid = [np.array(x, 'd') for x in np.mgrid[:texsize, :texsize]]
    uf = u_grid.flatten()
    vf = v_grid.flatten()
    for j in xrange(len(face_idx)):
      p1 = mesh_pts[face_idx[j][0]]
      p2 = mesh_pts[face_idx[j][1]]
      p3 = mesh_pts[face_idx[j][3]]

      e1 = -p1 + p3
      e2 = -p1 + p2

      n = ut.normalized(np.cross(e1, e2))
      d = -np.dot(p1, n) #np.dot(p1, n) + d = 0
      self.face_planes.append(np.concatenate([n, [d]]))
      
      pts = p1 + (uf/(texsize - 1.))[:, na]*e1[na, :] + (vf/(texsize - 1.))[:, na]*e2[na, :]

      juv = np.zeros((len(uf), 3), 'l')
      juv[:, 0] = j
      juv[:, 1] = uf
      juv[:, 2] = vf
      self.face_juv.append(juv)
      self.face_pts.append(pts)
      self.face_edges.append((p1, np.array([e1, e2])))
      self.face_center.append(p1 + 0.5*e1 + 0.5*e2)

    self.face_center = np.array(self.face_center) 
    self.tex2juv = np.vstack(self.face_juv)
    self.juv2tex = np.zeros((len(self.face_idx), texsize, texsize), 'l')
    self.juv2tex[self.tex2juv[:, 0], self.tex2juv[:, 1], self.tex2juv[:, 2]] = range(len(self.tex2juv))
    self.texel_pts = np.vstack(self.face_pts)
    self.face_planes = np.array(self.face_planes)
    self.on_border = ut.lor((self.tex2juv[:, 1] == 0),
                            (self.tex2juv[:, 2] == 0),
                            (self.tex2juv[:, 1] == self.texsize-1),
                            (self.tex2juv[:, 2] == self.texsize-1))
    
    self.ntexels = len(self.tex2juv)

  def texel_visible(self, scan, frame):
    ray_juv, dist = self.backproject_im(scan, frame)
    assert len(set(ray_juv[ray_juv[:, :, 0] >= 0, 0])) <= 3

    face_visible = np.zeros(self.nfaces, 'bool')
    face_visible[np.array(ray_juv[ray_juv[:, :, 0] >= 0, 0], 'l')] = 1
     
    proj = np.array(scan.project(frame, self.texel_pts), 'l')
    ok = ig.pixels_in_bounds(ray_juv.shape, proj[:, 0], proj[:, 1])
    visible = np.zeros(len(self.texel_pts), 'bool')

    visible[ok] = face_visible[self.tex2juv[ok, 0]]

    return visible

  def index_as_juv(self, xs):
    assert xs.shape[0] == len(self.tex2juv)

    if np.ndim(xs) == 1:
      a = np.zeros(self.juv2tex.shape[:3], dtype = xs.dtype)
    else:
      a = np.zeros(self.juv2tex.shape[:3] + (xs.shape[1],), dtype = xs.dtype)
      
    a[self.tex2juv[:, 0], self.tex2juv[:, 1], self.tex2juv[:, 2]] = xs

    return a

  def index_as_flat(self, xs):
    return xs[self.tex2juv[:, 0], self.tex2juv[:, 1], self.tex2juv[:, 2]]
                 
  # render with antialiasing
  def render(self, scan, frame, texel_colors, antialias = True, alias_steps = ALIAS_SAMPLES, im = None, mask = None):
    #  def render(self, scan, frame, texel_colors, antialias = True, alias_steps = 1):
    if im is None:
      im = scan.im(frame)
    if np.ndim(frame) == 0:
      R = scan.R(frame)
      P = scan.P(frame)
      K = scan.K(frame)
      center = scan.center(frame)
    else:
      R, P, K, center = frame
      
    # multiple rays per pixel
    if 1:
      ov = np.array(ut.bbox2d(mvg.homog_transform(P, self.mesh_pts.T).T), 'd')
      ov[:2] = np.floor(ov[:2])
      ov[2:] = np.ceil(ov[2:])
      ov = map(int, ov)
      ov = ut.rect_im_intersect(im, ov)

      sigma = 0.25
      #sigma = 0.5/(alias_steps/2)
      xx, yy, cyx, sample_weights = [], [], [], []
      for cx in xrange(ov[0], ov[0] + ov[2]):
        for cy in xrange(ov[1], ov[1] + ov[3]):
          d = alias_steps/2
          for dxi in xrange(-d, d+1):
            for dyi in xrange(-d, d+1):
              # xx.append(cx + 0.5 + dx/max(0.00001, (d/2.)))
              # yy.append(cy + 0.5 + dy/max(0.00001, (d/2.)))
              dxf = 0.5*float(dxi)/max(0.00001, d)
              dyf = 0.5*float(dyi)/max(0.00001, d)
              xx.append(cx + dxf)
              yy.append(cy + dyf)
              sample_weights.append(np.exp((-dxf**2 - dyf**2)/(2.*sigma**2)))
              
              #print float(-dxi**2 - dyi**2)/(2*1**2), (-dxf**2 - dyf**2)/(2.*sigma**2)
              #print np.exp((-dxi**2 - dyi**2)/(2*1**2)), sample_weights[-1]
              cyx.append([cy, cx])
      xx = np.array(xx, 'd')
      yy = np.array(yy, 'd')
      sample_weights = np.array(sample_weights, 'd')
      #flat_idx = np.ravel_multi_index(ut.ensure_col(np.array(cyx).T, 2), im.shape[:2])
      flat_idx = np.ravel_multi_index(ut.ensure_col(cyx, 2, 'l').T, im.shape[:2])
                       
    rays = np.dot(mvg.pixel_ray_matrix(R, K),
                  np.array([xx, yy, np.ones_like(xx)])).T

    if len(rays) == 0:
      return im
    
    ray_juv, _ = self.backproject_rays(center, rays)

    colors_juv = self.index_as_juv(texel_colors)
    ray_colors = np.zeros_like(ray_juv)

    for j in [-1] + range(self.nfaces):
      ok = (ray_juv[:, 0] == j)
      if j == -1:
      # no intersection; use the background
        ray_colors[ok] = ig.lookup_bilinear(im, xx[ok], yy[ok])
      else:
        #print np.min(ray_juv[ok, 2]), np.max(ray_juv[ok, 2])
        ray_colors[ok] = ig.lookup_bilinear(colors_juv[j], ray_juv[ok, 2], ray_juv[ok, 1])
        #ray_colors[ok] = colors_juv[j][np.array(np.round(ray_juv[ok, 1]),'l'), np.array(np.round(ray_juv[ok, 2]), 'l')]

    assert not np.any(np.isnan(ray_colors))
    
    # average together each pixel's samples
    color_sum = np.zeros(im.shape)
    for c in xrange(im.shape[2]):
      #color_sum[:, :, c] = np.bincount(flat_idx, weights = ray_colors[:, c], minlength = im.shape[0]*im.shape[1]).reshape(im.shape[:2])
      color_sum[:, :, c] = np.bincount(flat_idx, weights = ray_colors[:, c]*sample_weights, minlength = im.shape[0]*im.shape[1]).reshape(im.shape[:2])

    #counts = np.bincount(flat_idx, minlength = im.shape[0]*im.shape[1]).reshape(im.shape[:2])
    counts = np.bincount(flat_idx, weights = sample_weights, minlength = im.shape[0]*im.shape[1]).reshape(im.shape[:2])
    res_im = im.copy()
    filled = (counts != 0)
    res_im[filled] = color_sum[filled]/counts[filled][:,na]
    res_im[-filled] = im[-filled]

    # anti-aliasing will slightly blur things, even for pixels that don't actually see the box
    # fix this by setting all off-cube pixels equal to their original color
    hit_count = np.bincount(flat_idx, weights = (ray_juv[:, 0] >= 0), minlength = im.shape[0]*im.shape[1]).reshape(im.shape[:2])
    res_im[hit_count == 0] = im[hit_count == 0]
    
    #ray_count = np.bincount(flat_idx, minlength = im.shape[0]*im.shape[1]).reshape(im.shape[:2])

    if mask is not None:
      res_im = res_im*(1.-mask[:,:,np.newaxis]) + im*mask[:,:,np.newaxis]
      
    return res_im

  def backproject_rays(self, center, rays):
    # todo: verify that this is necesary
    rays = rays / ut.normax(rays, 1)[:, na]
    
    def backproj(face_visible):
      # (alpha*r + t)*n + d = 0
      # alpha*r'n + t'*n + d = 0
      # (-d - t'*n)/(r'*n)
      ray_juv = -1 + np.zeros(rays.shape, 'd')
      best_dist = np.inf + np.zeros(rays.shape[0])

      if len(rays) == 0:
        return ray_juv, best_dist
      
      for j in np.nonzero(face_visible)[0]:
        plane = self.face_planes[j]
        dist = (-plane[3] - np.dot(center, plane[:3]))/np.dot(rays, plane[:3])
        #dist2 = (-plane[3] - np.dot(center, plane[:3]))/np.dot(rays, plane[:3])
        pts = dist[:, na]*rays + center

        p1, edges = self.face_edges[j]
        # R = edges / ut.normax(edges, 1)[:, na]
        # uv = np.array((self.texsize-1)*np.dot(R/ut.normax(edges, 1)[:, na], (pts - p1).T), 'd').T

        uv = np.linalg.lstsq(edges.T, (pts - p1).T)[0].T

        in_bounds = ut.land(0 <= uv[:, 0], uv[:, 0] <= 1,
                            0 <= uv[:, 1], uv[:, 1] <= 1)

        uv = np.array((self.texsize-1)*uv)

        visible = ut.land(in_bounds, 0 <= dist, dist < best_dist)

        best_dist[visible] = dist[visible]
        ray_juv[visible, 0] = j
        ray_juv[visible, 1] = uv[visible, 0]
        ray_juv[visible, 2] = uv[visible, 1]

      #print np.bincount(np.array(ray_juv[ray_juv[:, 0] >= 0, 0], 'l'), minlength = 6)
      # in first call to backproj this assertion might fail (need to double check the rest of the code if that happens)
      assert np.sum(np.bincount(np.array(ray_juv[ray_juv[:, 0] >= 0, 0], 'l'), minlength = 6) > 0) <= 3

      return ray_juv, best_dist

    # potential problem: it's ambiguous what happens at an occlusion boundary, i.e. the places where the two faces meet
    # hacky solution: mark faces for which only edge pixels are visible, and mark them as occluded
    # I've never actually seen this problem happen, though...
    ray_juv, best_dist = backproj(np.ones(self.nfaces))
    visible = np.zeros(self.nfaces)
    rerun = False
    for j in xrange(self.nfaces):
      ok = ray_juv[:, 0] == j
      if np.sum(ok) == 0:
        visible[j] = False
      else:
        # only the border texels are visible; mark is invisible
        visible[j] = (not np.all(ut.lor(ray_juv[ok, 1] == 0, ray_juv[ok, 1] == self.texsize-1, ray_juv[ok, 2] == 0, ray_juv[ok, 2] == self.texsize-1)))
        # need to rerun (with the face marked as invisible)
        if not visible[j]:
          rerun = True
          print 'rerunning!'
          
    if rerun:
      ray_juv, best_dist = backproj(visible)
    
    return ray_juv, best_dist

  def backproject_im(self, scan, frame):
    shape = scan.im(frame).shape
    ray_dirs = ut.mult_im(scan.Rs[frame].T, mvg.ray_directions(scan.Ks[frame], shape))
    if 1:
      rays = ut.col_from_im(ray_dirs)
    if 0:
      #x, y = map(int, scan.project(frame, np.mean(self.mesh_pts, axis = 0)))
      x, y = map(int, scan.project(frame, self.mesh_pts[0]))
      rays = ray_dirs[y, x][na, :]
      print scan.project(frame, scan.center(frame) + rays[0]), (x, y)
    
    ray_juv, dist = self.backproject_rays(scan.center(frame), rays)
    #ray_juv = ray_juv.reshape((3, shape[0], shape[1])).transpose([1, 2, 0])
    ray_juv = ut.im_from_col(shape, ray_juv)
    dist = ut.im_from_col(shape[:2], dist)
    return ray_juv, dist

def mask(scan, mesh, frame):
  ray_juv, dist = mesh.backproject_im(scan, frame)
  return ray_juv[:, :, 0] >= 0

def faces_visible(scan, mesh, frame):
  ray_juv, dist = mesh.backproject_im(scan, frame)
  js = np.array(ray_juv[:, :, 0].flatten(), 'l')
  js = js[js >= 0]
  return np.bincount(js, minlength = 6) > 0

def load_from_mat(fname, texsize = None):
  m = scipy.io.loadmat(fname)
  if 'faces' in m:
    if np.ndim(m['faces']) == 2 and m['faces'].shape[1] == 4:
      face_idx = -1 + np.array(m['faces'], 'l')
    else:
      face_idx = -1 + np.array([m['faces'][0][i][0].flatten() for i in xrange(len(m['faces'][0]))], 'l')
  else:
    face_idx = -1 + np.array([[1, 2, 4, 3], np.array([1, 2, 4, 3])+4, [1, 2, 2+4, 1+4], [2, 4, 4+4, 2+4], [4, 3, 3+4, 4+4], [3, 1, 1+4, 3+4]])
    
  #face_pts = np.array([m['world_pos'][i][0].flatten() for i in xrange(len(m['world_pos']))], 'd')

  if np.ndim(m['world_pos']) == 2 and m['world_pos'].shape[1] == 3:
    face_pts = np.array(m['world_pos'], 'd')
  else:
    face_pts = np.array([m['world_pos'].squeeze()[i].flatten() for i in xrange(len(m['world_pos'].squeeze()))], 'd')
    
  return Mesh(face_idx, face_pts, texsize = texsize)

def mesh_from_path(path):
  return load_from_mat(ut.pjoin(path, 'cube.mat'))

def save_mesh(fname, mesh):
  scipy.io.savemat(fname, {'world_pos' : mesh.mesh_pts, 'faces' : 1+mesh.face_idx})
   
def draw_faces(mesh, scan, frame, hires = True, im = None, label_faces = False):
  face_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]], 'd')
  #face_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [255, 255, 0]], 'd')
  assert np.max(mesh.tex2juv[:, 0]) < face_colors.shape[0]
  colors = face_colors[mesh.tex2juv[:, 0]]

  if label_faces:
    as_juv = mesh.index_as_juv(colors)
    for j in xrange(mesh.nfaces):
      as_juv[j] = ig.draw_text(as_juv[j], str(j), [(0, 0)], [(255, 255, 255)], font_size = mesh.texsize)
      
      #ig.show(as_juv[j])
    colors = mesh.index_as_flat(as_juv)
      
  alias = ALIAS_SAMPLES if hires else 1
  return mesh.render(scan, frame, colors, alias, im = im)

def test_box(path, inds = None, hires = 1, as_cycle = False, label_faces = False, mesh = None):
  scan = dset.Scan(path)
  if mesh is None:
    mesh = load_from_mat(os.path.join(path, 'cube.mat'))
  ut.toplevel_locals()
  if inds is None:
    inds = range(scan.length)
  ims = [draw_faces(mesh, scan, i, hires = hires, label_faces = label_faces) for i in inds]

  if as_cycle:
    ig.show([('cycle', ims)])
  else:
    ig.show(ims)
  #ig.show([draw_faces(mesh, scan, i) for i in xrange(scan.length)])

def render_scene(scene, texel_colors, mesh = None):
  scan = dset.Scan(scene)
  if mesh is None:
    mesh = load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
  # click to toggle between the rendered image and the mesh
  ig.show([('cycle', [mesh.render(scan, frame, texel_colors), scan.im(frame)]) for frame in xrange(scan.length)])
  
  #ig.show([mesh.render(scan, frame, texel_colors) for frame in xrange(scan.length)])
