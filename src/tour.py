import util as ut, glob, os, dset, box, img as ig, experiments, numpy as np, scipy.io, camo, copy, rotation, mvg, imtable, iputil as ip, pylab, planefit, glob

import figures
    
def tour(scan, mesh, texel_colors, frames, n = 40, im_wait = 2.2,
         plane_idx = 2, par = 0, outline_start = np.inf, outline_end = np.inf, bad_pairs = [], other_planes = [],
         mesh_occ = None, start_scale = 1.25, start_wo_outline = False):
#def tour(scan, mesh, texel_colors, frames, n = 40, im_wait = 2.2, plane_idx = 2, par = 0, bad_pairs = [], other_planes = [], mesh_occ = None):
  if 0:
    idx = mesh.face_idx
    idx = idx[plane_idx:plane_idx+1]
    only_plane = box.Mesh(idx, mesh.mesh_pts)
    ig.show(box.draw_faces(only_plane, scan, 0))
    return

  if mesh_occ is None:
    mesh_occ = figures.MeshOcc(scan, None)
    
  def f(frame1, frame2, n = n, im_wait = im_wait, plane_idx = plane_idx, texel_colors = [texel_colors], scan = scan, mesh = mesh,
        other_planes = other_planes, bad_pairs = bad_pairs, frames = frames, mesh_occ = mesh_occ, outline_start = outline_start,
        outline_end = outline_end, start_scale = start_scale, start_wo_outline = start_wo_outline):
    texel_colors = texel_colors[0]
    
    ims = []
    ps = np.linspace(0, 1, n)

    frame1_idx = frames.index(frame1)
    frame2_idx = frames.index(frame2)


    for pi, p in enumerate(ps):
      print p
      
      p1 = 1-p
      p2 = p

      R1, c1 = scan.R(frame1), scan.center(frame1)
      R2, c2 = scan.R(frame2), scan.center(frame2)

      Rp = slerp_R(R1, R2, p)

      cp = p1*c1 + p2*c2
      Kp = p1*scan.K(frame1) + p2*scan.K(frame2)
      tp = -np.dot(Rp, cp)
      Pp = mvg.compose_P(Kp, Rp, tp)

      tc = texel_colors.copy()
      if outline_start <= frame1_idx < outline_end:# or outline_start <= frame2_idx < outline_end:
          if frame2_idx == outline_end:
            interp_occ = p1
          else:
            interp_occ = 1.

          print 'interp_occ =', interp_occ
          
          scan_with_interp, frame_interp = scan.add_virtual_camera(Kp, Rp, dset.t_from_Rc(Rp, cp), scan.im(frame1))

          frame_round = (frame1 if p <= 0.5 else frame1)
          tc = texel_colors.copy()
          tc = figures.mark_occlusion_texels(tc, scan_with_interp, mesh, frame_interp, thresh = 2,
                                             p = interp_occ, mesh_occ_mask = np.array(np.round(mesh_occ.mask(frame_round)), 'bool'))

        
      if pi == 0 or pi == len(ps)-1:
        frame = (frame1 if pi == 0 else frame2)
        #tc = figures.mark_occlusion_texels(tc, scan, mesh, frame, mesh_occ_mask = np.array(np.round(mesh_occ.mask(frame)), 'bool'), thresh = 2) 

        if frame == frames[0]:
          # 2x brings it on par with a normal frame (which are counted twice); add another factor to
          # give people a chance to scan
          wait_time = int(start_scale*2*im_wait*n)
        elif frame == frames[-1]:
          wait_time = int(2*im_wait*n)
        else:
          wait_time = int(im_wait*n)

        if start_wo_outline:
          ims += ([mesh.render(scan, frame, texel_colors)]*3)
        im = mesh.render(scan, frame, tc)
        im0 = scan.im(frame)
        ims += [mesh_occ.apply_mask(im, im0, mesh_occ.mask(frame))]*wait_time
        
      else:
        im1 = scan.im(frame1)
        im2 = scan.im(frame2)

        planes = [mesh.face_planes[plane_idx]] + other_planes

        #print other_planes
        best_dists = None
        for plane in planes:
          # backproject pixel into both frames, using the plane as the geometry
          # average the colors
          im = np.zeros_like(im1)

          #print plane
          ray_dirs = ut.col_from_im(ut.mult_im(Rp.T, mvg.ray_directions(Kp, im.shape)))
          dists = (-plane[3] - np.dot(cp, plane[:3]))/np.dot(ray_dirs, plane[:3])

          if best_dists is None:
            best_dists = dists
          else:
            # asdf 
            #best_dists[dists < best_dists] = 0
            if 1:
              ok = ((best_dists < 0) & (dists >= 0)) | ((dists >= 0) & (dists < best_dists))
              best_dists[ok] = dists[ok]
            
          pts = cp + ray_dirs*best_dists[:, np.newaxis]
          
        proj1 = scan.project(frame1, pts)
        proj2 = scan.project(frame2, pts)

        color1 = ig.lookup_bilinear(im1, proj1[:, 0], proj1[:, 1])
        color2 = ig.lookup_bilinear(im2, proj2[:, 0], proj2[:, 1])

        in_bounds1 = ig.lookup_bilinear(np.ones(im1.shape[:2]), proj1[:, 0], proj1[:, 1]) > 0
        in_bounds2 = ig.lookup_bilinear(np.ones(im2.shape[:2]), proj2[:, 0], proj2[:, 1]) > 0
        
        p1s = np.array([p1]*len(proj1))
        p1s[-in_bounds1] = 0

        p2s = np.array([p2]*len(proj2))
        p2s[-in_bounds2] = 0

        s = p1s + p2s
        p1s = p1s / np.maximum(s, 0.00001)
        p2s = p2s / np.maximum(s, 0.00001)

        #mask = p1*mesh_occ.mask(frame1) + p2*mesh_occ.mask(frame2)
        #mask = p1*mesh_occ.mask(frame1) + p2*mesh_occ.mask(frame2)
        mask1 = ig.lookup_bilinear(mesh_occ.mask(frame1), proj1[:, 0], proj1[:, 1])
        mask2 = ig.lookup_bilinear(mesh_occ.mask(frame2), proj2[:, 0], proj2[:, 1])
        mask = ut.im_from_col(im.shape[:2], mask1*p1s + mask2*p2s)
        
        #mask = p1*mesh_occ.mask(frame1) + p2*mesh_occ.mask(frame2)
        #mask[mask >= 0.5] = 1

        if (frame1, frame2) in bad_pairs:
          assert mesh_occ.path is None
          ims.append(p1*mesh.render(scan, frame1, tc, mask = mask) + p2*mesh.render(scan, frame2, tc, mask = mask))
        else:
          im = ut.im_from_col(im.shape, color1*p1s[:, np.newaxis] + color2*p2s[:, np.newaxis])
          ims.append(mesh.render(scan, (Rp, Pp, Kp, cp), tc, im = im, mask = mask))

    return ims

  ims = ut.flatten(ip.map(par, f, (frames[:-1], frames[1:])))

  #ut.toplevel_locals()
  
  #ig.show([('animation', ims, 100*10./n)])
  #ig.show([('animation', ims, 8*10./n)])
  #ig.show([imtable.Video(ims, fps = 0.25*float(im_wait*n))])
  #ig.show([imtable.Video(ims, fps = 0.75*float(im_wait*n))])

  ut.toplevel_locals()
  url = ig.show([imtable.Video(ims, fps = 0.75*float(im_wait*n))])
  
  del ims
  return url


def slerp_R(R1, R2, p):
  def r2q(R):
    A = np.eye(4)
    A[:3,:3] = R
    return rotation.quaternion_from_matrix(A)
  
  def q2r(q):
    return rotation.quaternion_matrix(q)[:3,:3]
  
  return q2r(rotation.quaternion_slerp(r2q(R1), r2q(R2), p))

def test_tour(par=1):
  ip.reset(par)
  scene = 'disrupt-11'
  scan = dset.Scan('../data/%s' % scene)
  texel_colors = ut.load(ut.pjoin(figures.make_path('noloo', 'interior-wide', scene), 'data.pk'))['ret'][0]
  mesh = box.load_from_mat(ut.pjoin(scan.path, 'cube.mat'))
  tour(scan, mesh, texel_colors, [0, 1, 2], plane_idx = 0, outline_start = 0, par = par)

