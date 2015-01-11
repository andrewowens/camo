import dset, planefit, numpy as np, pylab, img as ig, util as ut, mvg, box, time, os, glob

def label_box(seq, root = 0, side_len1 = None, side_len2 = None, side_len3 = None, y_flip = True, mode = 'normal'):
  print seq

  if type(seq) == type(''):
    scan = dset.Scan(seq, None)
  else:
    scan = seq
    seq = scan.path
      
  if mode == 'normal':
    _, _, tracks = dset.read_bundler(scan.bundle_file, scan.full_shape)
    pts = np.array([t[0] for t in tracks])

    proj = scan.project(root, pts)

    w = 1
    
    pylab.clf()
    im_with_pts = ig.draw_pts(scan.im(root), proj, width = w)
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

    im_ins = ig.draw_pts(scan.im(root), map(ut.itup, proj), map(ut.itup, colors), width = w)
    pylab.clf()
    pylab.imshow(im_ins)

    if not input('ok? '):
      return
    
    print 'click 2 points (used to recalibrate the plane)'
    rect = ut.bbox2d(pylab.ginput(2, timeout = -1))
    ok = ut.land(rect[0] <= proj[:, 0], proj[:, 0] <= rect[0] + rect[2], rect[1] <= proj[:, 1], proj[:, 1] <= rect[1] + rect[3])
    pts_in_box = pts[ok]
    print 'plane before', plane
    plane[3] = -np.median(np.dot(pts_in_box, plane[:3]))
    print 'plane after', plane[3]

    if 1:
      print 'hack'
      im_ins = scan.im(root)
    
    pylab.clf()
    pylab.imshow(im_ins)
    print 'click 3 base points'
    px = pylab.ginput(3, timeout = -1)
    #px = [(2270.2989175686921, 1482.9937552039967), (2297.2764363030801, 1555.8330557868442), (2405.1865112406322, 1550.4375520399667)]

    def backproj(p):
      ray = ut.normalized(np.dot(mvg.pixel_ray_matrix(scan.R(root), scan.K(root)), ut.homog(p)))
      c = scan.center(root)

      dist = (-plane[3] - np.dot(c, plane[:3]))/np.dot(ray, plane[:3])
      assert dist >= 0
      pt = c + ray*dist
      print planefit.dist_to_plane(plane, pt[np.newaxis, :])
      return pt

    sc = 1.
    while 1:
      cb = np.array(map(backproj, px))
      v1 = cb[0] - cb[1]
      v2 = cb[2] - cb[1]

      if side_len1 is None:
        side_len1 = 0.5*(np.linalg.norm(v1) + np.linalg.norm(v2))
      if side_len2 is None:
        side_len2 = side_len1
      if side_len3 is None:
        side_len3 = side_len1

      a1 = sc*side_len1
      a2 = sc*side_len2
      a3 = sc*side_len3

      print 'side1', a1, 'side2', a2, 'side3', a3, 'thresh =', thresh, \
            'v1 =', np.linalg.norm(v1), 'v2 = ', np.linalg.norm(v2)

      R = np.zeros((3, 3))
      cr = ut.normalized(np.cross(v1, plane[:3]))
      cr *= np.sign(np.dot(cr, v2))
      R[0] = a1*ut.normalized(v1)
      R[1] = a2*ut.normalized(cr)
      R[2] = a3*ut.normalized(plane[:3])
      print ut.normax(R, 1)

      mesh_pts = []
      for zi in xrange(2):
        for yi in xrange(2):
          for xi in xrange(2):
            mesh_pts.append(cb[1] + R[0]*xi + R[1]*yi + R[2]*zi)
      face_idx = -1 + np.array([[1, 2, 4, 3], np.array([1, 2, 4, 3])+4, [1, 2, 2+4, 1+4], [2, 4, 4+4, 2+4], [4, 3, 3+4, 4+4], [3, 1, 1+4, 3+4]])
      mesh = box.Mesh(face_idx, mesh_pts, texsize = 128)

      # show a preview
      scan_ = dset.Scan(seq)
      ig.show([[1+i, box.draw_faces(mesh, scan_, i, hires = 0), scan_.im(i)] for i in [root, root+1]])
      if input('ok? '):
        box.save_mesh(ut.pjoin(seq, 'cube.mat'), mesh)
        break
      else:
        sc = float(input('scale? '))
        time.sleep(2)

  else:
    mesh = box.load_from_mat(ut.pjoin(seq, 'cube.mat'))
  
  scan = dset.Scan(seq, use_cams_file = False)

  print 'Already marked as bad:'
  good_cams_file = os.path.join(scan.path, 'good_cams.txt')
  if os.path.exists(good_cams_file):
    inds = map(int, open(good_cams_file, 'r').read().split())
    file_ids = map(scan.file_index, scan.im_files)
    bad = sorted(set(file_ids) - set(inds))
    print '\n'.join(map(str, bad))
    
  if 1:
    ig.show([[scan.file_index(scan.im_files[frame]), box.draw_faces(mesh, scan, frame, hires = 0)] for frame in xrange(scan.length)])

  inp = input('Bad cameras (as string): ')
  if inp != 'skip':
    bad_cams = map(int, inp.split())
    all_idx = map(scan.file_index, scan.im_files)
    good_cams = sorted(set(all_idx) - set(bad_cams))
    ut.write_lines(ut.pjoin(seq, 'good_cams.txt'), map(str, good_cams))

def label_util1():
  scan = dset.Scan('../data/util1-real')
  scan.im_files[3] = os.path.join(scan.path, 'raw/2014-05-24 17.34.10.jpg')
  scan.im_files[4] = os.path.join(scan.path, 'raw/2014-05-24 17.35.07.jpg')
  label_box(scan, 3)
  mesh = box.load_from_mat('../data/util1-real/cube.mat')
  ig.show([box.draw_faces(mesh, scan, 3, hires = 0), scan.im(3)])

def label_util2():
  scan = dset.Scan('../data/util2-real')
  #ig.show([[frame, ig.resize(scan.im(frame), 0.5), f, ig.resize(ig.load(f), scan.scale*0.5)] for frame, f in zip(scan.frames, glob.glob(ut.pjoin(scan.path, 'raw', '*.jpg')))])

  frame = 0
  scan.im_files[frame] = '/data/vision/billf/camo/camo/nondetect/data/util2-real/raw/2014-05-25 17.17.02.jpg'

  label_box(scan, frame)

  mesh = box.load_from_mat('../data/util2-real/cube.mat')
  ig.show([box.draw_faces(mesh, scan, frame), scan.im(frame)])

def label_util3():
  scan = dset.Scan('../data/util3-real')
  #ig.show([[frame, ig.resize(scan.im(frame), 0.5), f, ig.resize(ig.load(f), scan.scale*0.5)] for frame, f in zip(scan.frames, glob.glob(ut.pjoin(scan.path, 'raw', '*.jpg')))])
  #return

  # frame = 21
  # scan.im_files[frame] = '/data/vision/billf/camo/camo/nondetect/data/util3-real/raw/2014-05-25 18.02.06.jpg'
  frame = 25

  label_box(scan, frame)

  mesh = box.load_from_mat('../data/util-real/cube.mat')
  ig.show([box.draw_faces(mesh, scan, frame), scan.im(frame)])
