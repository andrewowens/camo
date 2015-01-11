import util as ut, glob, os, dset, box, img as ig, experiments, numpy as np, scipy.io, camo, copy, rotation, mvg, imtable, iputil as ip, pylab, planefit, glob, tour

# Code for generating the figures/videos in the paper and the talk

RESDIR_LOO = '/data/scratch/aho/camo-results/camera-ready-loo/loo'
RESDIR_NOLOO = '/data/scratch/aho/camo-results/camera-ready-noloo/noloo'

STATS_PATH = '/data/vision/billf/camo/camo/nondetect/results/stats/stats.pk'

ALL_SCENES = experiments.classic_scenes + experiments.new_scenes

def make_path(loo_s, alg_name, scene = ''):
  assert loo_s in ('loo', 'noloo')
  base = RESDIR_LOO if loo_s == 'loo' else RESDIR_NOLOO

  return ut.pjoin(base, idx_from_alg(alg_name), scene)

def idx_from_alg(alg_name):
  return str(METHODS.index(alg_name)+1)
  
def path_from_scene(scene):
  return ut.pjoin('../data', scene)

# duplicated from 4-8 in experiments.py
METHODS = ['uniform', 'mean', 'random', 'greedy', 'occlusion', 'stable-robust', 'occlusion-wide', 'interior-wide', 'occlusion-wide-nostable']
tested_scenes = experiments.classic_scenes + experiments.new_scenes

def make_teaser():
  name = 'bookshelf-real'
  in_dir = make_path('noloo', 'interior-wide', name)
  out_dir = '../results/teaser-bookshelf-interior'
  ut.mkdir(out_dir)
  print in_dir
  for in_fname in sorted(glob.glob(ut.pjoin(in_dir, '*.jpg'))):
    base = os.path.split(in_fname)[1]
    out_fname = ut.pjoin(out_dir, base.replace('.jpg', '.pdf'))
    assert not os.path.exists(out_fname)
    print in_fname, out_fname
    os.system('convert %s %s' % (in_fname, out_fname))

def make_scene_fig(nscenes = 20, ims_per_scene = 2, todo = ['fig']):
  with ut.constant_seed(0):
    method = 'occlusion-wide'
    # random sample
    #all_scenes = sorted(glob.glob(make_path('noloo', method, '*')))
    #scenes = ut.sample_at_most(all_scenes, nscenes)

    #already_in_paper = 'couch3-real bookshelf-real'.split()
    already_in_paper = ''.split()
    
    ok_scenes = []
    # these mess up the diagram
    for scene in tested_scenes:
      shape = dset.Scan(path_from_scene(scene)).full_shape
      ratio = (float(shape[1]) / float(shape[0]))
      if abs(ratio  - 1.5 ) >= 0.01 or (scene in already_in_paper):
        print 'skipping', scene, 'bad aspect ratio', ratio, 'or already in paper'
      else:
        ok_scenes.append(scene)

    #scenes = ut.sample_at_most(ut.shuffled(scenes), nscenes)
    scenes = ut.sample_at_most(ut.shuffled(ok_scenes), nscenes)

    print '\n'.join(scenes)

    if 'show' in todo:
      table = []
      for scene in scenes:
        print scene
        scan = dset.Scan(path_from_scene(scene))
        mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
        texel_colors = ut.load(ut.pjoin(make_path('noloo', method, scene), 'data.pk'))['ret'][0]

        row = [scene]
        assert ims_per_scene == 2
        # show frames from this result
        # choose two that aren't used in the solution and which are representative viewpoints
        # this is nontrivial to do programmatically; pick them by hand
        # include in the UI a way to verify that the same image is not being used
        # note that due to the sampling some views might be of faces that have no label (check this!)
        for frame in scan.frames:
          row += [frame, ('cycle', [mesh.render(scan, frame, texel_colors), scan.im(frame)])]
        table.append(row)

      ig.show(table)

    if 'fig' in todo:
      frame_choices = \
                    {'mit-31' : 11,
                     'mit-29' : 0,
                     'disrupt-8' : 12,
                     'mit-12': 15,
                     'patio2-real' : 1,
                     'walden-tree1' : 9,
                     'mit-12' : 19,
                     'mit-21' : 8,
                     'charlottesville-6' : 6,
                     'walden-log' : 6,
                     'charlottesville-2' : 8,
                     'charlottesville-9' : 6,
                     'charlottesville-1' : 7,
                     'disrupt-6' : 0,
                     'mit-20' : 3,
                     'mit-14': 13,
                     'walden-tree3' : 0,
                     'mit-6' : 6,
                     'mit-1' : 8,
                     'mit-5' : 16,
                     'couch3-real' : 6,
                     'bookshelf-real' : 3,
                     'charlottesville-7' : 9,
                     'mit-26' : 8,
                     'mit-28' : 13,
                     'mit-13' : 7,
                     'disrupt-11' : 7,
                     'couch5-real' : 2,
                     'walden-brush2' : 0,
                     'mit-9' : 0,
                     'mit-27' : 0,
                     'charlottesville-3' : 1,
                     'mit-37' : 4,
                     'mit-16' : 13,
                     }

      out_base = '../results/scene-fig'
      #assert not os.path.exists(out_base)
      ut.mkdir(out_base)

      scene_acc = ut.load(STATS_PATH)

      scenes_by_easiness = sorted(scenes, key = lambda x : -np.mean(scene_acc[x, idx_from_alg(method)]))
      
      for si, scene in enumerate(scenes_by_easiness):
        print scene, np.mean(scene_acc[scene, idx_from_alg(method)])

        # easier than deriving the image number from the output files
        scan = dset.Scan(path_from_scene(scene))
        texel_colors = ut.load(ut.pjoin(make_path('noloo', method, scene), 'data.pk'))['ret'][0]
        frame = frame_choices[scene]
        #out_path = ut.pjoin(out_base, 'scene-%d.pdf' % (1+si))
        out_path = ut.pjoin(out_base, 'scene-%d.png' % (1+si))
        #assert not os.path.exists(out_path)
        mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
        ig.save(out_path, mesh.render(scan, frame, texel_colors))
      
def make_multiview_fig(n = None):
  method = 'occlusion-wide'
  scene_choices = ['mit-1', 'charlottesville-1', 'disrupt-11']
  frame_choices = {'mit-1' : [0, 3, 7, 10], 'charlottesville-1' : [0, 2, 5, 8], 'disrupt-11' : [0, 4, 7, 10]}

  out_base = '../results/multiview-fig'
  #assert not os.path.exists(out_base)
  ut.mkdir(out_base)
  
  for si, scene in enumerate(scene_choices[:n]):
    scan = dset.Scan(path_from_scene(scene))
    mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
    frames = frame_choices[scene]
    for fi, frame in enumerate(frames):
      out_path = ut.pjoin(out_base, 'scene-%d-%d.png' % (1+si, 1+fi))
      #assert not os.path.exists(out_path)
      texel_colors = ut.load(ut.pjoin(make_path('noloo', method, scene), 'data.pk'))['ret'][0]
      
      #ig.save(out_path, mesh.render(scan, frame, texel_colors))
      #ig.save(out_path, render_cube(scan.path, mesh, texel_colors, frame, 200, outline = True, frame_using_cube = True))
      if scene == 'mit-1':
        occ_thresh = 1.25#1.8
      else:
        occ_thresh = None

      if scene == 'charlottesville-1':
        occ_thresh = 2.5
        d_sc = 1.2
      else:
        d_sc = 1.
        
      ig.save(out_path, render_cube(scan.path, mesh, texel_colors, frame, 200, outline = True, frame_using_cube = True, occ_thresh = occ_thresh, dim_sc = d_sc))

  # def dilate_occ(scan, mesh, frame):
  #   occ = camo.occlusion_texels(scan, mesh, frame, thresh = 1.5, only_border = True)
  #   as_juv = mesh.index_as_juv(occ).copy()

  #   for j in xrange(as_juv.shape[0]):
  #     #dist, ind = scipy.ndimage.distance_transform_edt(1 - as_juv[j], return_indices = True)
  #     if np.any(as_juv[j]):
  #       dist, ind = scipy.ndimage.distance_transform_bf(1 - as_juv[j], metric = 'taxicab', return_indices = True)
  #       dist[ind[0] < 0] = 1e10
  #       as_juv[j, dist <= 10] = True

  #   return np.logical_and(mesh.texel_visible(scan, frame), mesh.index_as_flat(as_juv))



def clean_occ(scan, mesh, frame):
  occ = camo.occlusion_texels(scan, mesh, frame, thresh = 1.5, only_border = True)
  as_juv = mesh.index_as_juv(occ).copy()

  for j in xrange(as_juv.shape[0]):
    w, h = as_juv.shape[1:]
    for u, v in [(0, range(h)),
                 (range(w), 0),
                 (range(w), -1),
                 (-1, range(h))]:
      as_juv[j, u, v] = (np.mean(as_juv[j, u, v]) >= 0.5)
      
    #dist, ind = scipy.ndimage.distance_transform_edt(1 - as_juv[j], return_indices = True)
    # if np.any(as_juv[j]):
    #   dist, ind = scipy.ndimage.distance_transform_bf(1 - as_juv[j], metric = 'taxicab', return_indices = True)
    #   dist[ind[0] < 0] = 1e10
    #   as_juv[j, dist <= 10] = True

  return np.logical_and(mesh.texel_visible(scan, frame), mesh.index_as_flat(as_juv))


def scan_fullres(fr, path):
  if fr:
    return dset.Scan(path, max_dim = None)
  else:
    return dset.Scan(path)

  
def occlusion_mask(scan, mesh, frame, thresh = 2., outline = False):
  mask = box.mask(scan, mesh, frame)
  #D = scipy.ndimage.distance_transform_edt(mask)
  D = scipy.ndimage.distance_transform_edt(mask)
  return D <= thresh, D
  #return np.logical_and(mask, D <= thresh)


def mark_occlusion_texels(tc, scan, mesh, frame, thresh, mesh_occ_mask = None, p = 1):
  tc = tc.copy()

  mask = box.mask(scan, mesh, frame)
  if mesh_occ_mask is not None:
    mask = (mask & -mesh_occ_mask)
    
  D = scipy.ndimage.distance_transform_edt(mask)
  
  #occ_mask = np.array(occlusion_mask(scan, mesh, frame, thresh = thresh), 'd')
  occ_mask = np.array(D, 'd')
    
  vis = mesh.texel_visible(scan, frame)
  proj = scan.project(frame, mesh.texel_pts)
  proj = np.array(np.round(proj), 'l')
  occ = np.zeros(mesh.ntexels, 'd')
  occ[vis] = occ_mask[proj[vis, 1], proj[vis, 0]]

  w = np.zeros_like(occ)
  w[occ < thresh] = p#1
    
  # scale the texels that are not totally on the boundary
  ok = (thresh <= occ) & (occ < 1+thresh)
  # anti-alias and (optionally) weight
  w[ok] = p*((1+thresh) - occ[ok])
  assert np.all((0 <= w) & (w <= 1))
  tc = tc*(1-w[:, np.newaxis]) + 255*w[:, np.newaxis]
  return tc


def render_cube(scene, mesh, texel_colors, frame, crop_size, fullres = False, outline = False,
                frame_using_cube = False, occ_thresh = None, draw_boundaries = False, im = None, use_fr = True, dim_sc = 1., show_cube = True):
  scan = scan_fullres(fullres, scene)
    
  if im is None:
    im_input = scan.im(frame)
  else:
    im_input = im

  tc = texel_colors.copy()

  mask = box.mask(scan, mesh, frame)
  ys, xs = np.nonzero(mask)
  cx, cy = map(int, np.mean(np.array([xs, ys]), axis = 1))

  if frame_using_cube:
    box_rect = ut.bbox2d(np.array([xs, ys]).T)
    d = int(round(dim_sc * min(4*max(box_rect[2:]), min(scan.im(0).shape[:2]) - 1)))
    rect = ut.rect_centered_at(cx, cy, d, d)
    rect = ut.shift_in_bounds(scan.im(0).shape, rect)
    scale = float(crop_size)/rect[2]
    print box_rect, rect, scale
  else:
    rect = None
    scale = 1.

  # if not show_cube:
  #   im = scan.im(frame)
  #   rect = ut.rect_centered_at(cx, cy, crop_size, crop_size)
  #   crop_size /= scan_fullres(False, scan.path).scale
  #   return ig.sub_img(im, ut.rect_im_intersect(im, rect))
  
  if outline:
    if rect is not None:
      assert rect[2] == rect[3]
    #scan_fr = scan_fullres(True, scene)
    scan_fr = scan_fullres(use_fr, scene)
    print 'scale', scale

    if occ_thresh is None:
      occ_thresh = 2.
    occ_thresh /= scale

    # occ = camo.occlusion_texels(scan_fr, mesh, frame, thresh = occ_thresh, only_border = False)
    # tc[occ] = 255

    
    tc = mark_occlusion_texels(tc, scan_fr, mesh, frame, thresh = occ_thresh)
    
    im_up = ig.resize(im_input, scan_fr.im(frame).shape)
    #im = ig.resize(mesh.render(scan_fr, frame, tc, im = im_up), scan.im(frame).shape)
    im_fr = mesh.render(scan_fr, frame, tc, im = im_up)
    im = ig.resize(im_fr, scan.im(frame).shape)
    if not show_cube:
      im = scan.im(frame)
    #ig.show([im_fr, im])
    #assert im.shape[0] == im.shape[1]
  else:
    if show_cube:
      im = mesh.render(scan, frame, tc)
    else:
      im = scan.im(frame)
    
  
  if rect is not None:
    if draw_boundaries:
      return ig.draw_rects(im, [rect])
    else:
      return ig.sub_img(im, rect)
  if fullres:
    crop_size /= scan_fullres(False, scan.path).scale
    #sc = (crop_size/2.)/float(box_rect[2])
  elif crop_size is None:
    return im
  else:
    rect = ut.rect_centered_at(cx, cy, crop_size, crop_size)
    return ig.sub_img(im, ut.rect_im_intersect(im, rect))
  
def make_real_cube():
  scene = 'bookshelf-real'
  method = 'interior-wide'
  scan = dset.Scan(path_from_scene(scene))

  #texel_colors = camo.to_color_space_2d(ut.load(ut.pjoin(make_path('noloo', method, scene), 'data.pk'))['ret'][0])
  if 0:
    print 'hires'
    mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'), texsize = 512)
    scan = dset.Scan(path_from_scene(scene), max_dim = 2000)
    texel_colors = camo.camo(scan, mesh, ut.Struct(method = 'interior-wide'))
    ut.save('../results/real-interior.pk', texel_colors)
    ut.toplevel_locals()
  elif 1:
    # upgrade to larger texel size; bigger images
    texel_colors0, results0, labels0 = ut.load(ut.pjoin(make_path('noloo', method, scene), 'data.pk'))['ret'][:3]
    mesh0 = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'), texsize = 256)
    texsize = 1024
    mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'), texsize = texsize)
    geom = camo.Geom(scan, mesh)
    scan = dset.Scan(path_from_scene(scene), max_dim = 2000)
    label_color, label_valid = camo.label_colors(scan, mesh, geom, labels0, invisible_colors = True)

    as_juv0 = mesh0.index_as_juv(results0).copy()
    as_juv1 = mesh.index_as_juv(np.zeros(mesh.ntexels)).copy()

    for j in xrange(as_juv0.shape[0]):
      as_juv1[j] = ig.resize(as_juv0[j], as_juv1[j].shape[:2], order = 0, hires = False)
    results1 = np.array(mesh.index_as_flat(as_juv1), 'l')

    texel_colors = camo.from_color_space_2d(label_color[range(len(results1)), results1])
    #texel_colors = label_color[range(len(results1)), results1]

    ut.toplevel_locals()
  elif 0:
    texel_colors = ut.load('../results/real-interior.pk')[0]
    mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'), texsize = 512)
  else:
    mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
    texel_colors = ut.load(ut.pjoin(make_path('noloo', method, scene), 'data.pk'))['ret'][0]
    
  texel_colors = camo.to_color_space_2d(texel_colors)
  texel_colors = mesh.index_as_juv(texel_colors)
  out_path = '../results/real/colors.mat'
  scipy.io.savemat(out_path, {'texel_colors' : texel_colors})
  import matlab
  matlab.matlab_cmd('/data/vision/billf/camo/camo', 'load_real_cube')

def make_printable_pattern(scene_path, mesh0, texel_colors0, results0, labels0, geom = None):
  #texel_colors = camo.to_color_space_2d(ut.load(ut.pjoin(make_path('noloo', method, scene), 'data.pk'))['ret'][0])
  # upgrade to larger texel size; bigger images
  #texsize = 1024
  texsize = 4096
  #scan = dset.Scan(scene_path, max_dim = 2000)
  scan = dset.Scan(scene_path, max_dim = None)
  mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'), texsize = texsize)
  

  as_juv0 = mesh0.index_as_juv(results0).copy()
  as_juv1 = mesh.index_as_juv(np.zeros(mesh.ntexels)).copy()

  for j in xrange(as_juv0.shape[0]):
    as_juv1[j] = ig.resize(as_juv0[j], as_juv1[j].shape[:2], order = 0, hires = False)
    
  results1 = np.array(mesh.index_as_flat(as_juv1), 'l')

  labels = np.array(labels0, 'double')
  labels[:, 1:] *= scan.scale/dset.Scan(scene_path).scale

  print labels

  texel_colors = np.zeros((mesh.ntexels, 3))

  if geom is None:
    geom = camo.Geom(scan, mesh)
    
  print len(np.unique(results1))
  for label in np.unique(results1):
    print 'trying', label
    label = int(label)
    frame = int(labels[label, 0])
    valid, colors = camo.project_texels(scan, frame, mesh, scan.im(frame), geom, labels[label, 1:])
    ok = results1 == label
    texel_colors[ok] = colors[ok]
                              
  #texel_colors = label_color[range(len(results1)), results1]

  texel_colors_rgb = texel_colors.copy()
  
  ut.toplevel_locals()

  #texel_colors = camo.to_color_space_2d(texel_colors)
  texel_colors = mesh.index_as_juv(texel_colors)
  out_path = '../results/real/colors.mat'
  scipy.io.savemat(out_path, {'texel_colors' : texel_colors, 'texel_colors_rgb' : texel_colors_rgb})


  # import matlab
  # matlab.matlab_cmd('/data/vision/billf/camo/camo', 'load_real_cube')


def make_rescomp_fig(n = None):
  table = []
  # index frames to be consistent w/ amt results

  comparisons = [
    ('mit-20', 3, ['occlusion-wide', 'interior-wide']),
    ('disrupt-14', 4, ['occlusion-wide', 'random']),
    ('disrupt-14', 5, ['occlusion-wide', 'random']),
    ('disrupt-14', 3, ['occlusion-wide', 'random']),
    ]

  #scene_acc = ut.load(STATS_PATH)

  #out_dir = '../results/qual-compare'
  out_dir = '../results/qual-compare2'
  ut.mkdir(out_dir)
  num = 0
  for scene, loo_idx, methods in comparisons:
    scan = scan_fullres(False, path_from_scene(scene))
    mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
    table.append([])

    for method in methods:
      data = ut.load(ut.pjoin(make_path('loo', method, scene), 'data_%03d.pk' % loo_idx))
      texel_colors = data['ret'][0]
      loo_frame = scan.idx.index(data['pr'].loo_frame_idx)

      im = render_cube(scan.path, mesh, texel_colors, loo_frame, 200, outline = True, frame_using_cube = True)
      assert im.shape[0] == im.shape[1]
      table[-1] += [method, im]
      #table[-1] += [method, render_cube(scan.path, mesh, texel_colors, loo_frame, 200, outline = False)]
      ig.save(ut.pjoin(out_dir, 'result-%03d.pdf' % num), im)
      ig.save(ut.pjoin(out_dir, 'result-%03d.png' % num), im)
      num += 1
  ig.show(table)

def count_ims():
  total = 0
  for scene in ALL_SCENES:
    path = make_path('loo', 'interior-wide', scene)
    nims = len(glob.glob(path + '/result_*.jpg'))
    total += nims
    print scene, nims
  print 'total images', total, 'scenes', len(ALL_SCENES)
    

def draw_grid(im, proj, spacing = [-1, 0, 1]):
  d = 30.
  for x in spacing:
    for y in spacing:
      if x < 1:
        im = ig.draw_lines(im, [proj + d*np.array([x, y])], [proj + d*np.array([x+1, y])], colors = (255, 255, 255))
      if y < 1:
        im = ig.draw_lines(im, [proj + d*np.array([x, y])], [proj + d*np.array([x, y+1])], colors = (255, 255, 255))
  return im
  
def make_project_fig():
  #scene = 'mit-35'
  scene = 'mit-37'
  path = make_path('noloo', 'interior-wide', scene)
  #texel_colors = ut.load(ut.pjoin(path, 'data.pk'))['ret'][0]
      
  scan = dset.Scan(path_from_scene(scene))
  mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
  scan = dset.Scan(path_from_scene(scene))
  frames = range(scan.length)#[scan.length-1] #range(scan.length)[:1]

  geom = camo.Geom(scan, mesh)

  #texel = mesh.juv2tex[5, 128, 128]
  texel = mesh.juv2tex[2, 128, 128]
  
  table = []
  for frame in frames:
    proj = scan.project(frame, mesh.texel_pts[texel])
    im_grid = draw_grid(scan.im(frame), proj)
    label_valid, self_colors = camo.project_texels(scan, frame, mesh, im_grid, geom)
    im = render_cube(scan.path, mesh, self_colors, frame, 200, fullres = False, outline = True, frame_using_cube = True, occ_thresh = 2., draw_boundaries = True, im = im_grid, use_fr = False)
    table.append([im, scan.im(frame)])
    #table.append(ig.draw_pts(im, proj))

  ig.show(table)
    

def find_best_algs():
  with ut.constant_seed():
    scene_acc = ut.load(STATS_PATH)
    for scene in ALL_SCENES:
      algs = ['greedy', 'interior-wide', 'occlusion-wide']
      acc = [np.mean(scene_acc[scene, idx_from_alg(alg)]) for alg in algs]
      i = np.argmin(acc)
      #print scene, algs[i], acc
      yield scene, algs[i]

def label_plane(seq, root = 0, y_flip = True):
  scan = dset.Scan(seq, None)
  _, _, tracks = dset.read_bundler(scan.bundle_file, scan.full_shape)
  pts = np.array([t[0] for t in tracks])
  
  proj = scan.project(root, pts)

  pylab.clf()
  im_with_pts = ig.draw_pts(scan.im(root), proj, width = 2)
  pylab.imshow(im_with_pts)
  rect = ut.bbox2d(pylab.ginput(2, timeout = -1))
  #rect = (1782.005828476269, 1431.7364696086595, 529.75936719400488, 354.40549542048279)
  print rect

  ok = ut.land(rect[0] <= proj[:, 0], proj[:, 0] <= rect[0] + rect[2], rect[1] <= proj[:, 1], proj[:, 1] <= rect[1] + rect[3])
  pts_in_box = pts[ok]
  thresh = pylab.dist(scan.center(root), scan.center(root+1))/50.
  plane, _ = planefit.fit_plane_ransac(pts_in_box, thresh)
  if plane[1] < 0 and y_flip:
    plane *= -1

  ins = planefit.plane_inliers(plane, pts, thresh)

  pylab.clf()
  colors = np.zeros_like(pts)
  colors[:, 0] = 255
  colors[ins] = (0, 255, 0)

  im_ins = ig.draw_pts(scan.im(root), map(ut.itup, proj), map(ut.itup, colors), width = 2)
  pylab.clf()
  pylab.imshow(im_ins)
  
  print plane
  return plane

video_order = ['charlottesville-3', 'bookshelf-real', 'disrupt-11', 'mit-14', 'walden-brush2', 'walden-log', \
               'disrupt-8', 'charlottesville-1', 'mit-13', 'disrupt-6']
        
def test_warp(par = 0, start = 0, end = None):
  #scenes = ['bookshelf-real']#['couch3-real', 'bookshelf-real', 'disrupt-11', 'patio2-real', 'mit-1', 'disrupt-8', 'charlottesville-2']
  # 5: side
  # 4: usually side
  # 3: usually side
  # 2: usually side
  # 1: usually top
  # 0: usually bottom
  # np.array([-0.9286861 ,  0.13738529, -0.34448136, -3.96361632])
  scenes = [('disrupt-11', 0, 1, [], []),
            ('charlottesville-2', 0, 1, [], [(8, 9)]),
            ('mit-27', 0, 1, [np.array([ -9.06738777e-01,   2.58900135e-03,   4.21684821e-01, 2.93683015e+00])], []),
            ('disrupt-6', 0, 1, [np.array([ 0.85136312,  0.18874681, -0.48944405, -1.52800028])], []),
            ('couch3-real', 0, 1, [np.array([-0.60995728,  0.15168697, -0.77778094, -0.88194374])], []),
            ('couch5-real', 2, 1, [], []),
            ('disrupt-8', 0, 1, [np.array([-0.92784247,  0.1387372 , -0.34620851, -3.97233358])], []),
            ('mit-13', 0, -1, [], []),
            ('mit-20', 0, -1, [], []),
            ('bookshelf-real', 3, -1, [], [])]
#            ('disrupt-6', 0, 1, [], [np.array([ 0.85139516,  0.190946  , -0.48853444, -1.52601666])]),
  for x in ALL_SCENES:
    if x not in map(ut.fst, scenes):
      scenes.append((x, 0, 1, [], []))
  #print scenes

  #scenes = scenes[start:end]
  scenes = sorted(scenes, key = lambda x : (len(video_order) if x[0] not in video_order else video_order.index(x[0]), x[0]))
  scenes = scenes[start:end]
  ip.reset(par)

  scene_alg = dict(find_best_algs())

  #scene_names = [y[0] for y in scenes]
  for scene, plane_idx, order, other_planes, bad_pairs in scenes:
    #texel_colors = ut.load(ut.pjoin(make_path('noloo', 'interior-wide', scene), 'data.pk'))['ret'][0]
    alg = 'random' #scene_alg[scene]
    texel_colors = ut.load(ut.pjoin(make_path('noloo', alg, scene), 'data.pk'))['ret'][0]
    scan = dset.Scan(ut.pjoin('../data/', scene))
    mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
    #tour.tour(scan, mesh, texel_colors, [0, 1, 2, 3], par = par)
    if 0:
      if order == 1:
        frames = range(scan.length)
      else:
        frames = list(reversed(range(scan.length)))
    else:
      frames = sorted(set(map(int, np.linspace(0, scan.length-1, 6))))
      if order != 1:
        frames = list(reversed(frames))
    
    #print 'before mem usage'
    #ut.mem_usage()
    print scene, alg
    # url = tour.tour(scan, mesh, texel_colors, frames, plane_idx = plane_idx,
    #                 other_planes = other_planes, bad_pairs = bad_pairs,
    #                 outline_start = 0, outline_end = 1, start_wo_outline = True, par = par)
    
    #url = tour.tour(scan, mesh, texel_colors, frames, plane_idx = plane_idx, other_planes = other_planes, bad_pairs = bad_pairs, outline_start = scan.length/2, par = par)
    url = tour.tour(scan, mesh, texel_colors, frames, plane_idx = plane_idx, other_planes = other_planes, bad_pairs = bad_pairs, par = par)
    f = open('../results/vid-list', 'a')
    print >>f, scene, alg, url
    f.close()
    #print other_planes
    #url = tour.tour(scan, mesh, texel_colors,  [scan.length-2, scan.length-1], n = 5, plane_idx = plane_idx, other_planes = other_planes, par = par)
    #print 'after mem usage'
    #ut.mem_usage()

def make_warps():
  for i in xrange(len(ALL_SCENES)):
    os.system('python -c "import figures; figures.test_warp(par = 1, start = %d, end = %d+1)"' % (i, i))

def collect_warps():
  urls = [x.split() for x in ut.lines('../results/vid-results')]
  base = '/data/vision/billf/aho-billf/www/tab'
  out = ut.make_temp_dir(dir = base)
  f = open(ut.pjoin(out, 'index.html'), 'w')
  for _, _, url in urls:
    last = url.split('/')[-1]
    path = os.path.join(base, last)
    page_in = open(ut.pjoin(path, 'index.html'), 'r')
    f.write(page_in.read() + '\n')
    for y in glob.glob(path + '/*.mp4'):
      os.system('ln -s %s %s/' % (y, out))
  f.close()
  os.system('chmod -R a+rwx %s' % out)
  print ut.pjoin(imtable.PUBLIC_URL, out.split('/')[-1])

class MeshOcc:
  def __init__(self, scan, mask_path = None):
    self.scan = scan
    self.path = mask_path

  def mask(self, frame):
    if self.path is None:
      return np.zeros(self.scan.im(frame).shape[:2])
    else:
      fname = os.path.join(self.path, 'masked%d.png'% (frame+1))
      if os.path.exists(fname):
        mask = np.all(ig.load(fname) == (255, 0, 255), axis = 2)
        mask = 255*np.array(mask, 'd')
        mask = ig.resize(mask, self.scan.scale, hires = 1)/255.
        return mask
      else:
        return np.zeros(self.scan.im(frame).shape[:2])

  def apply_mask(self, im_mesh, im_nomesh, mask):
    return im_mesh*(1.-mask[:,:,np.newaxis]) + im_nomesh*mask[:,:,np.newaxis]
    
def make_videos():
  # order = ['charlottesville-3', 'bookshelf-real', 'disrupt-11', 'mit-14', 'walden-brush2', 'mit-27', 'mit-1', 'walden-log', \
  #          'mit-5', 'charlottesville-1', 'couch3-real',  'disrupt-6', 'disrupt-8', 'mit-13']
  print 'tmp'
  video_order = ['charlottesville-3', 'bookshelf-real', 'disrupt-11', 'mit-14', 'walden-log', \
                 'disrupt-8', 'charlottesville-1', 'mit-13', 'disrupt-6']
  vids = []
  #urls = dict([(x.split()[0], x.split()[1:]) for x in ut.lines('../results/vid-results')])
  urls = dict([(x.split()[0], x.split()[1:]) for x in ut.lines('../results/vid-list')])
  base = '/data/vision/billf/aho-billf/www/tab'
  for scene in video_order:
    alg, url = urls[scene]
    print 'alg', alg
    last = url.split('/')[-1]
    path = os.path.join(base, last)
    vids.append(glob.glob(path + '/*.mp4')[0])

  print '\n'.join(vids)
  ut.write_lines('../results/ffmpeg-vid-list', ['file %s' % s for s in vids])
  os.system('ffmpeg -f concat -i ../results/ffmpeg-vid-list  -c copy /data/vision/billf/aho-billf/www/camo-vid.mp4')


  
def make_nondetect_slide(todo, par = False):
  ip.reset(par)
  scene = 'bookshelf-real'
  #scan = dset.Scan(ut.pjoin('../data/', scene))
  #scan = dset.Scan(ut.pjoin('../data/', scene), max_dim = 500.)
  scan = dset.Scan(ut.pjoin('../data/', scene))
  mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))

  frame = 6
  
  # fix the mesh
  lf = frame-1

  if 'move' in todo:
    # front
    plane1 = mesh.face_planes[5]
    # side
    plane2 = mesh.face_planes[4]
    pt1 = mesh.mesh_pts[mesh.face_idx[lf][0]]
    pt2 = mesh.mesh_pts[mesh.face_idx[lf][1]]

    table = []
    # for d in np.linspace(0.8,  0.9, 5):
    #   for d2 in np.linspace(0.05,  0.5, 5):

    for d in [0.85]:
      for d2 in [1-0.85]:
        #for d3 in [0.1, 0.11, 0.12, 0.15]:
        for d3 in [0.15]:
          for d4 in [0, 0.025, 0.05, 0.1]:

            if 1:
              bottom = 1
              top = 0

              mesh_pts = mesh.mesh_pts.copy()

              for i in xrange(scan.length):
                if i in (top, bottom):
                  pts = mesh.mesh_pts[mesh.face_idx[i]].copy()
                  c = np.mean(pts, axis = 0)
                  mesh_pts[mesh.face_idx[i]] = c + d*(pts - c)

                  if i == top:
                    mesh_pts[mesh.face_idx[i]] -= d2*pylab.dist(pt1, pt2)*mesh.face_planes[i][:3]

              #mesh2 = box.Mesh(mesh.face_idx, mesh_pts)

            #mesh2 = box.Mesh(mesh.face_idx, mesh_pts - plane[:3]*0.07*pylab.dist(pt1, pt2))
            mesh2 = box.Mesh(mesh.face_idx, mesh_pts - plane1[:3]*d3*pylab.dist(pt1, pt2))
            mesh2 = box.Mesh(mesh.face_idx, mesh_pts - plane2[:3]*d4*pylab.dist(pt1, pt2))
            table.append([d, d2, d3, d4] +  [box.draw_faces(mesh2, scan, i) for i in scan.frames])
    ig.show(table)
  
  #frame = 3



  #lf = 1
  lf = frame-1
  d = np.linalg.norm(scan.center(lf) - scan.center(lf-1))
  pt = scan.center(lf) + 1.05*d*np.array([0., 1., 0]) #0.1*mesh.face_planes[-1][:3]*d
  im = scan.im(lf)

  #VDIR = mvg.ray_dirs(scan.K(lf), im.shape, scan.R(lf))[im.shape[0]/2, im.shape[1]/2]

  texel_colors = np.zeros((mesh.ntexels, 3))
  for face in xrange(6):
    print np.abs(np.dot(ut.normalized(-mesh.face_center[face] + pt), ut.normalized(mesh.face_planes[face][:3])))
    texel_colors[mesh.tex2juv[:, 0] == face] = 255*np.abs(np.dot(ut.normalized(-mesh.face_center[face] + pt),
                                                                  ut.normalized(mesh.face_planes[face][:3])))

  #texel_colors *= 225/float(np.max(texel_colors))
  texel_colors *= 255/float(np.max(texel_colors))

  lighting_colors = texel_colors.copy()
  ut.save('../results/bookshelf-lighting.pk', lighting_colors)

  mesh_occ = MeshOcc(scan, '../results/bookshelf-masks')

  def mesh_render(*args, **kwargs):
    kwargs['mask'] = mesh_occ.mask(args[1])
    return mesh.render(*args, **kwargs)
    
  if 'lighting' in todo:
    ig.show([[mesh_render(scan, f, texel_colors), scan.im_with_real(f)] for f in range(scan.length)])

  geom = camo.Geom(scan, mesh)
  
  if 'random' in todo:
    #for other_frame in xrange(scan.length):
    table = []
    for frame1 in scan.frames:
      _, texel_colors = camo.project_texels(scan, frame1, mesh, scan.im(frame1), geom)
      table.append([])
      for frame2 in scan.frames:
        table[-1] += [frame1, frame2, mesh_render(scan, frame2, texel_colors)]
    ig.show(table)

  if 'tour-random' in todo:
    #frames = [6, 0]
    #frames = [6, 2]
    frames = [6, 3]
    valid, proj_colors = camo.project_texels(scan, frames[0], mesh, scan.im(frames[0]), geom)
    texel_colors = lighting_colors.copy()
    texel_colors[valid] = proj_colors[valid]

    tour.tour(scan, mesh, texel_colors, frames, plane_idx = 3, par = par)

  if 'distortion-real' in todo:
    src_frame = 2
    view_frame = 1
    face = 2
    scan_tour = dset.Scan(ut.pjoin('../data/', scene))
    colors = lighting_colors.copy()
    colors[:] = 200
    colors[mesh.tex2juv[:, :, 0] == face] = (0, 128, 0)

    table = []
    valid, proj_colors = camo.project_texels(scan_tour, src_frame, mesh, scan_tour.im(src_frame), geom)
    colors[valid] = proj_colors
    table.append(mesh_render(scan_tour, view_frame, colors))
    
    ig.show(table)


  if 'distortion-synthetic' in todo:
    proj_frame = 2
    view_frame = 1
    
    im = scan.im(proj_frame).copy()
    mask = box.mask(scan, mesh, proj_frame)

    #pattern = ig.load('/data/vision/billf/camo/camo/nondetect/results/textures/zebra-stripes-vector/zebra-stripes.png')
    #pattern = ig.load('/data/vision/billf/camo/camo/nondetect/results/textures/checkers/Checkerboard_pattern.png')
    pattern = ig.load('/data/vision/billf/camo/camo/nondetect/results/textures/checkers/Checkerboard_pattern.jpg')
    #pattern = pattern.transpose([1, 0, 2])

    ys, xs = np.nonzero(mask)
    rect = ut.bbox2d(zip(xs, ys))

    s = 1.02*max(float(rect[3]) / pattern.shape[0], float(rect[2]) / pattern.shape[1])

    pattern = ig.resize(pattern, s)

    cx, cy = map(int, np.mean(np.array([xs, ys]), axis = 1))
    ig.sub_img(im, ut.rect_centered_at(cx, cy, pattern.shape[1], pattern.shape[0]))[:] = pattern

    _, texel_colors = camo.project_texels(scan, proj_frame, mesh, im, geom)
    texel_colors = texel_colors * np.array(lighting_colors, 'd')/255.    

    table = []
    # table.append(mesh_render(scan, view_frame, texel_colors, im = 255+np.zeros_like(im)))
    # table.append(mesh_render(scan, proj_frame, texel_colors, im = 255+np.zeros_like(im)))
    table.append(mesh_render(scan, view_frame, texel_colors))
    table.append(mesh_render(scan, proj_frame, texel_colors))
    ig.show(table)
    
  if 'tour-cues' in todo:
    #ntour = 5
    ntour = 40
    do_tours = False
    
    #scan_tour = dset.Scan(ut.pjoin('../data/', scene), max_dim = 500.)
    scan_tour = dset.Scan(ut.pjoin('../data/', scene))
    frames = [6, 2]
    valid, proj_colors = camo.project_texels(scan_tour, frames[0], mesh, scan_tour.im(frames[0]), geom)
    texel_colors = 0.75*lighting_colors.copy()
    texel_colors[valid] = proj_colors[valid]

    print 'distortion and occlusion boundary cues'
    if do_tours: tour.tour(scan_tour, mesh, texel_colors, frames, plane_idx = 3, im_wait = 1, n = ntour,
              mesh_occ = mesh_occ, outline_start = 0, outline_end = 1, par = par)

    table = []
    # all
    table.append(mesh_render(scan_tour, frames[-1], texel_colors))

    sc = 0.4
    im_dark = sc*scan_tour.im(frames[-1])
    table.append(sc*mesh_render(scan_tour, frames[-1], texel_colors))
    ig.show(table)
    table.append(mesh_render(scan_tour, frames[-1], texel_colors, im = im_dark))

    # distortion and occlusion
    #for f in [0, 6]:#xrange(6):
    for f in xrange(6):#xrange(6):
      tc = texel_colors.copy()
      tc[mesh.tex2juv[:, 0] != f] *= sc
      table.append([f, mesh_render(scan_tour, frames[-1], tc, im = im_dark)])
    ig.show(table)
    
    print "what happens if we look at a view that wasn't covered?"
    frames2 = [2, 0]
    ig.show(mesh_render(scan_tour, frames[-1], tc, im = im_dark))
    if do_tours: tour.tour(scan_tour, mesh, texel_colors, frames2, n = ntour, plane_idx = 3, im_wait = 1, par = par, mesh_occ = mesh_occ)

    print 'we can fill it with something...'
    other_frame = 1
    valid2, proj_colors2 = camo.project_texels(scan_tour, other_frame, mesh, scan_tour.im(other_frame), geom)

    texel_colors_filled = texel_colors.copy()
    texel_colors_filled[(-valid) & valid2] = proj_colors2[-valid & valid2]
 
    im_dark = sc*scan_tour.im(frames2[-1])
    ig.show([mesh_render(scan_tour, frames2[-1], texel_colors_filled),
             mesh_render(scan_tour, frames2[-1], texel_colors_filled, im = im_dark)])
    
    ig.show([mesh_render(scan_tour, f, texel_colors_filled) for f in scan.frames])
      
    
    #if do_tours: tour.tour(scan_tour, mesh, texel_colors, frames, plane_idx = 3, im_wait = 1, par = par)


  if 'test-mask' in todo:
    table = []
    #scan = scan_fr = scan_fullres(True, scan.path)
    for frame in scan.frames:
      fname = '../results/bookshelf-masks/im%d-colored.png' % (frame+1)
      if os.path.exists(fname):
        mask = np.all(ig.load(fname) == (255, 0, 255), axis = 2)
        mask = 255*np.array(mask, 'd')
        mask = ig.resize(mask, scan.scale, hires = 1)/255.
        #mask = np.array(ig.resize(np.array(mask, 'd'), scan.scale, order = 1, hires = 0))
        #ig.show(mask)
        im = box.draw_faces(mesh, scan, frame)
        if 0:
          im[mask] = scan.im(frame)[mask]
        if 1:
          im = im*(1.-mask[:,:,np.newaxis]) + mask[:,:,np.newaxis]*scan.im(frame)
        table.append(im)

    ig.show(table)
    
  if 'mask' in todo:
    scan_fr = scan_fullres(True, scan.path)
    ig.show([[scan_fr.im(f), box.draw_faces(mesh, scan_fr, f)] for f in scan_fr.frames])
    
    
  if 'zebra' in todo:
    table = []
    # for frame1 in scan.frames:
    #   for other_frame in scan.frames: #[scan.length-1]:
    for frame1 in [6]:
      for other_frame in [4]: #[scan.length-1]:
        im = scan.im(other_frame).copy()
        mask = box.mask(scan, mesh, other_frame)

        pattern = ig.load('/data/vision/billf/camo/camo/nondetect/results/textures/zebra-stripes-vector/zebra-stripes.png')
        pattern = pattern.transpose([1, 0, 2])

        ys, xs = np.nonzero(mask)
        #cx, cy = map(int, np.mean(np.array([xs, ys]), axis = 1))
        rect = ut.bbox2d(zip(xs, ys))

        s = 1.02*max(float(rect[3]) / pattern.shape[0], float(rect[2]) / pattern.shape[1])

        print s

        pattern = ig.resize(pattern, s)

        #ig.sub_img(im, rect)[:] = pattern
        cx, cy = map(int, np.mean(np.array([xs, ys]), axis = 1))
        ig.sub_img(im, ut.rect_centered_at(cx, cy, pattern.shape[1], pattern.shape[0]))[:] = pattern
        
        _, texel_colors = camo.project_texels(scan, other_frame, mesh, im, geom)

        texel_colors = texel_colors * np.array(lighting_colors, 'd')/255.

        table.append([frame1, other_frame, mesh_render(scan, frame1, texel_colors), mesh_render(scan, frame1, lighting_colors), scan.im_with_real(frame1)])

    ig.show(table)

def make_octopus_video():
  vids_in = ['../results/Ovulg_Wow_sequence_Silent_RTHWatermark.mov', '../results/Octopus Vulgaris.mov']
  #clips = ['../results/octo-clip1.mov', '../results/octo-clip2.mov']
  clips = ['../results/octo-clip1.mov', '../results/octo-clip2.mp4']
  times = ['-ss 00:00:00 -t 10', '-ss 00:05:48 -t 12.5']

  # for vid, clip, time in zip(vids_in, clips, times):
  #   os.system('ffmpeg -i "%s" %s "%s"' % (vid, time, clip))
      
  ut.write_lines('../results/ffmpeg-octopus-list', ['file %s' % s for s in clips])
  os.system('ffmpeg -f concat -i ../results/ffmpeg-octopus-list  -c copy ../results/octopus-talk.mp4')
  print 'scp', 'aho@vision11.csail.mit.edu:' + os.path.abspath('../results/octopus-talk.mp4'), '~/Dropbox/cvpr-talk/octopus-talk.mp4'
 

def spotlight_slides(par = 1):
  scene = 'charlottesville-3'
  alg = 'occlusion-wide'
  scan = dset.Scan(ut.pjoin('../data/', scene))
  mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
  #texel_colors = ut.load(ut.pjoin(make_path('noloo', alg, scene), 'data.pk'))['ret'][0]  
  #lighting_colors = ut.load('../results/bookshelf-lighting.pk')

  frame = 0
  lf = 4
  d = np.linalg.norm(scan.center(lf) - scan.center(lf-1))
  pt = scan.center(lf) + 2.05*d*np.array([0., 1., 0]) #1.05*d*np.array([0., 1., 0]) #0.1*mesh.face_planes[-1][:3]*d
  im = scan.im(lf)

  #VDIR = mvg.ray_dirs(scan.K(lf), im.shape, scan.R(lf))[im.shape[0]/2, im.shape[1]/2]

  texel_colors = np.zeros((mesh.ntexels, 3))
  for face in xrange(6):
    print np.abs(np.dot(ut.normalized(-mesh.face_center[face] + pt), ut.normalized(mesh.face_planes[face][:3])))
    texel_colors[mesh.tex2juv[:, 0] == face] = 255*np.abs(np.dot(ut.normalized(-mesh.face_center[face] + pt),
                                                                  ut.normalized(mesh.face_planes[face][:3])))
  texel_colors *= 255/float(np.max(texel_colors))
  lighting_colors = texel_colors

  im = mesh.render(scan, frame, lighting_colors)
  #ig.show(ig.resize(im, 0.5))
  texel_colors = ut.load(ut.pjoin(make_path('noloo', alg, scene), 'data.pk'))['ret'][0]
  im_colored = mesh.render(scan, frame, texel_colors)
  ig.show([im, im_colored])
  
  plane_idx = 0
  url = tour.tour(scan, mesh, texel_colors, range(scan.length), plane_idx = plane_idx, par = par, start_scale = 0)
  ig.show(url)
  
  
def make_occ_examples(texel_colors=None):
  ut.seed_rng(0)

  scene = 'walden-brush2'
  if 0:
    #alg = 'greedy'
    #texel_colors = ut.load(ut.pjoin(make_path('noloo', alg, scene), 'data.pk'))['ret'][0]
    #render_cube(scan.path, mesh, texel_colors, frame, 200, outline = True, frame_using_cube = True)
    scan_fr = dset.Scan(ut.pjoin('../data', scene), max_dim = None)
    mesh = box.load_from_mat(ut.pjoin(scan_fr.path, 'cube.mat'), 512)
    if texel_colors is not None:
      try:
        texel_colors = camo.camo(scan_fr, mesh, ut.Struct(method = 'greedy'))[0]
      except:
        print 'exception'

  ut.toplevel_locals()

  if 1:
    scan = dset.Scan(ut.pjoin('../data', scene))
    mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
    table = []
    view_frame = 2
    for frame in scan.frames:
      try:
        texel_colors = camo.camo(scan, mesh, ut.Struct(method = 'order', order = [frame]))[0]
        table.append([frame, render_cube(scan.path, mesh, texel_colors, view_frame, 200, fullres = True, outline = False, frame_using_cube = True)])
      except:
        print 'exception'
    ig.show(table)
    return
  
  frame = 2
  im_bg = render_cube(scan_fr.path, mesh, texel_colors, frame, 200, fullres = True, outline = True, frame_using_cube = True, show_cube = False)
  ig.show(im_bg)
  im_nooutline = render_cube(scan_fr.path, mesh, texel_colors, frame, 200, fullres = True, outline = False, frame_using_cube = True)
  im_outline = render_cube(scan_fr.path, mesh, texel_colors, frame, 200, fullres = True, outline = True, frame_using_cube = True)
  ig.show([im_bg, im_nooutline, im_outline])


def make_occlusion_slide():
  #scan = dset.Scan('../data/disrupt-14')
  #scan = dset.Scan('../data/walden-tree1')
  scan = dset.Scan('../data/mit-13', max_dim = None)
  #mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
  mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'), 1024)
  # table = []
  # for frame in scan.frames:
  #   view_frame = frame
  #   texel_colors = camo.camo(scan, mesh, ut.Struct(method = 'order', order = [frame]))[0]
  #   table.append([frame, render_cube(scan.path, mesh, texel_colors, view_frame, 200, fullres = True, outline = True, frame_using_cube = True)])
  # ig.show(table)

  #texel = mesh.juv2tex[2, 128, 128]
  #texel = mesh.juv2tex[2, 0, 128]
  #texel = mesh.juv2tex[2, 0, 128]
  texel = mesh.juv2tex[5, 0, 128]
  geom = camo.Geom(scan, mesh)
  
  table = []
  for frame in [0, 2, 4, 6]:#scan.frames:#[0, 1, 4]:
    #table.append(scan.im(frame))
    proj = scan.project(frame, mesh.texel_pts[texel])
    if 1:
      im_grid = draw_grid(scan.im(frame), proj, spacing = [0])
    else:
      im_grid = scan.im(frame)
    label_valid, self_colors = camo.project_texels(scan, frame, mesh, im_grid, geom)
    #im = render_cube(scan.path, mesh, self_colors, frame, 200, fullres = False, outline = True, frame_using_cube = True, occ_thresh = 2., draw_boundaries = True, im = im_grid, use_fr = False)
    im_nooutline = render_cube(scan.path, mesh, self_colors, frame, 200, fullres = True, outline = False, frame_using_cube = True)
    im_outline = render_cube(scan.path, mesh, self_colors, frame, 200, fullres = True, outline = True, frame_using_cube = True)
    #im = render_cube(scan.path, mesh, self_colors, frame, 200, fullres = True, outline = True, frame_using_cube = True, occ_thresh = 2., draw_boundaries = True, im = im_grid, use_fr = False)
    table.append([str(frame), im_outline, im_nooutline])
    
    #table.append([str(frame), im, scan.im(frame)])
    #table.append(ig.draw_pts(im, proj))

  ig.show(table)
  
def make_capture_slide():
  scene = 'mit-37'
  scan = dset.Scan(ut.pjoin('../data', scene))
  mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
  
  if 'lighting':
    lf = 13
    d = np.linalg.norm(scan.center(lf) - scan.center(lf-1))
    pt = scan.center(lf) + 2.05*d*np.array([0., 1., 0]) #1.05*d*np.array([0., 1., 0]) #0.1*mesh.face_planes[-1][:3]*d
    #im = scan.im(lf)

    #VDIR = mvg.ray_dirs(scan.K(lf), im.shape, scan.R(lf))[im.shape[0]/2, im.shape[1]/2]

    texel_colors = np.zeros((mesh.ntexels, 3))
    for face in xrange(6):
      print np.abs(np.dot(ut.normalized(-mesh.face_center[face] + pt), ut.normalized(mesh.face_planes[face][:3])))
      texel_colors[mesh.tex2juv[:, 0] == face] = 255*np.abs(np.dot(ut.normalized(-mesh.face_center[face] + pt),
                                                                    ut.normalized(mesh.face_planes[face][:3])))
    texel_colors *= 255/float(np.max(texel_colors))
    lighting_colors = texel_colors
    
  # geom = camo.Geom(scan, mesh)

  # #texel = mesh.juv2tex[5, 128, 128]
  # texel = mesh.juv2tex[2, 128, 128]
  frames = [8, 10, 13, 15]
  #ig.show([[mesh.render(scan, frame, lighting_colors), scan.im(frame)] for frame in frames])

  f1 = 10
  f2 = 14
  #f2 = 8
  geom = camo.Geom(scan, mesh)
  label_valid, tc1 = camo.project_texels(scan, f1, mesh, scan.im(f1), geom)
  label_valid, tc2 = camo.project_texels(scan, f2, mesh, scan.im(f2), geom)
  tc = tc1.copy()
  #ok = label_valid & (mesh.tex2juv[:, 0] == 2) & (mesh.tex2juv[:, 1] > 128)
  ok = (mesh.tex2juv[:, 0] == 1) | (mesh.tex2juv[:, 0] == 5) | (mesh.tex2juv[:, 0] == 3)
  ok = (mesh.tex2juv[:, 1] > 128)
  tc[ok] = tc2[ok]
  #ig.show(mesh.render(scan, f1, tc))

  vf = f1
  c1 = render_cube(scan.path, mesh, tc1, vf, 200, outline = True, frame_using_cube = True)
  c2 = render_cube(scan.path, mesh, tc2, vf, 200, outline = True, frame_using_cube = True)
  ch = render_cube(scan.path, mesh, tc, vf, 200, outline = True, frame_using_cube = True)
  ig.show([c1, c2, ch])
  return

  # c1 = mesh.render(scan, vf, tc1)
  # c2 = mesh.render(scan, vf, tc2)
  c1 = c2 = ''
  ch = mesh.render(scan, vf, tc)
  ig.show([c1, c2, ch])

  
# def make_camo_game():
  
#   tour.tour(scan_tour, mesh, texel_colors, frames, plane_idx = 3, im_wait = 1, n = ntour,
#             mesh_occ = mesh_occ, outline_start = 0, outline_end = 1, par = par)
