import numpy as np
cimport numpy as np
cimport cython

def prn(x):
  print x
  return

def solve_mrf(mrf):
  cdef np.ndarray[np.float32_t, ndim = 2] data_cost_ = np.ascontiguousarray(mrf.data_cost)
  cdef float *data_cost = <float*>(data_cost_.data)
  cdef a = prn('cost')
  #cdef np.ndarray[np.uint8_t, ndim = 3] label_color_ = np.ascontiguousarray(mrf.label_color)
  cdef np.ndarray[np.float32_t, ndim = 3] label_color_ = np.ascontiguousarray(mrf.label_color)
  cdef float *label_color = <float *>(label_color_.data)
  cdef int nnodes = mrf.data_cost.shape[0]
  cdef int nlabels = mrf.data_cost.shape[1]
  cdef np.ndarray[np.int32_t, ndim = 2] edges_ = np.array(mrf.edges, 'int32')
  cdef b = prn('edges')
  cdef int * edges = <int*>(edges_.data)
  cdef int nedges = edges_.shape[0]
  cdef np.ndarray[np.int32_t] labels_out = np.zeros(nnodes, 'int32')
  cdef int * labels = <int*>(labels_out.data)
  cdef np.ndarray[np.uint8_t] node_visible_ = np.array(mrf.node_visible, 'uint8')
  cdef c = prn('node_visible')
  cdef unsigned char * node_visible = <unsigned char*>(node_visible_.data)
  cdef np.ndarray[np.float32_t] sameview_prior_ = mrf.sameview_prior
  cdef float *sameview_prior = <float*>(sameview_prior_.data)
  cdef float smooth_prior = mrf.smooth_prior
  cdef d = prn('sameview')
    
  solve(data_cost, label_color, nlabels, nnodes, nedges, edges, node_visible, sameview_prior, smooth_prior, labels)

  return labels_out

def solve_face_mrf(mesh, mrf):
  # collapse each face into a supernode
  cdef int nlabels = mrf.data_cost.shape[1]
  cdef np.ndarray[np.float64_t, ndim = 2] fd = np.zeros((mesh.nfaces, nlabels), 'float64')
  cdef np.ndarray[np.float64_t, ndim = 2] ec = np.zeros((mesh.nfaces, mesh.nfaces), 'float64')
  cdef double ucost = 0, bcost = 0, best_cost = np.inf, best_ucost = np.inf, best_bcost = np.inf, c = 0
  cdef int label0, label1
  cdef int x0, x1, x2, x3, x4, x5
  cdef np.ndarray[np.int32_t] labels = np.zeros(mesh.nfaces, 'int32')
  
  for j in xrange(mesh.nfaces):
    fd[j] = np.sum(mrf.data_cost[mesh.tex2juv[:, 0] == j], axis = 0)

  face_edge_counts = np.zeros((fd.shape[0], fd.shape[1]))
  assert mrf.smooth_prior == 0
  for u, v in mrf.edges:
    if mrf.node_visible[u] and mrf.node_visible[v]:
      ec[mesh.tex2juv[u, 0], mesh.tex2juv[v, 0]] += 1
      ec[mesh.tex2juv[v, 0], mesh.tex2juv[u, 0]] += 1

  ec *= 1e5

  print 'pairwise costs'
  print np.array(ec, 'int64')
  
  best_cost = np.inf
  assert mesh.nfaces == 6
  for label0 in range(nlabels):
    for label1 in range(label0+1, nlabels):
      for label2 in range(label1+1, nlabels):
        assert label0 != label1 and label1 != label2 and label0 != label2
        for x0 in range(0, 3):
          for x1 in range(0, 3):
            for x2 in range(0, 3):
              for x3 in range(0, 3):
                for x4 in range(0, 3):
                  for x5 in range(0, 3):
                    y0 = (label0 if x0 == 0 else (label1 if x0 == 1 else label2))
                    y1 = (label0 if x1 == 0 else (label1 if x1 == 1 else label2))
                    y2 = (label0 if x2 == 0 else (label1 if x2 == 1 else label2))
                    y3 = (label0 if x3 == 0 else (label1 if x3 == 1 else label2))
                    y4 = (label0 if x4 == 0 else (label1 if x4 == 1 else label2))
                    y5 = (label0 if x5 == 0 else (label1 if x5 == 1 else label2))

                    ucost = 0.
                    ucost += fd[0, y0]
                    ucost += fd[1, y1]
                    ucost += fd[2, y2]
                    ucost += fd[3, y3]
                    ucost += fd[4, y4]
                    ucost += fd[5, y5]

                    bcost = ((y0 != y1)*ec[0, 1] + (y0 != y2)*ec[0, 2] + (y0 != y3)*ec[0, 3] + (y0 != y4)*ec[0, 4] + (y0 != y5)*ec[0, 5] + (y1 != y2)*ec[1, 2] + (y1 != y3)*ec[1, 3] + (y1 != y4)*ec[1, 4] + (y1 != y5)*ec[1, 5] + (y2 != y3)*ec[2, 3] + (y2 != y4)*ec[2, 4] + (y2 != y5)*ec[2, 5] + (y3 != y4)*ec[3, 4] + (y3 != y5)*ec[3, 5] + (y4 != y5)*ec[4, 5])

                    c = ucost + bcost
                    if c < best_cost:
                      best_cost = c
                      best_ucost = ucost
                      best_bcost = bcost
                      labels = np.array(np.array([label0, label1, label2])[np.array([x0, x1, x2, x3, x4, x5])], 'int32')

  print 'Total cost:', best_cost, 'Unary:', best_ucost, 'Binary:', best_bcost
  
  labels_node = labels[mesh.tex2juv[:, 0]]
  labels_node[-mrf.node_visible] = 0
  return labels_node

cdef extern from "mrf.h":
  # except + means: handle c++ exception
  void solve(float *data_cost, float *label_color, int nlabels, int nnodes, int nedges, int *edges,
           unsigned char *node_visible, float *sameview_prior, float smooth_prior, int *labels) except +

