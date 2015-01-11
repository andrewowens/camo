#include <algorithm>
#include <vector>
#include <stdio.h>
#include "GCoptimization.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <stack>
#include <ext/hash_map>

using namespace std;

typedef float en_t;
typedef unsigned char uchar;
typedef GCoptimization::EnergyTermType en_t;

struct Params {
  //uchar *label_color;
  float *label_color;
  uchar *node_visible;
  int nlabels, nnodes;
  float *sameview_prior;
  float smooth_prior;
};

Params *gl_p = NULL;

const int GC_ITERS = 5;
const int CHANNELS = 3;

inline float diff_sq(float x, float y) {
  float z = x - y;
  return z*z;
}
  
en_t smooth_cost(int n1, int n2, int label1, int label2) {
  assert(gl_p != NULL);
  Params &p = *gl_p;

  if (!p.node_visible[n1] || !p.node_visible[n2]) {
    return 0;
  } else {
    float cost = 0;
    //cost += p.sameview_prior*(label1 != label2);
    cost += max(p.sameview_prior[n1], p.sameview_prior[n2])*(label1 != label2);

   if ((p.smooth_prior > 0) && (label1 != label2)) {
      float *c1p = p.label_color + n1*(p.nlabels*CHANNELS) + label1*CHANNELS;
      float *c1q = p.label_color + n1*(p.nlabels*CHANNELS) + label2*CHANNELS;
      float *c2p = p.label_color + n2*(p.nlabels*CHANNELS) + label1*CHANNELS;
      float *c2q = p.label_color + n2*(p.nlabels*CHANNELS) + label2*CHANNELS;

      float dist = 0;
      for (int k = 0; k < 3; k++) {
        dist += diff_sq(c1p[k], c1q[k]) + diff_sq(c2p[k], c2q[k]);
      }
      cost += sqrt(dist);
    }
    
    return cost;
  }
}

void solve(float *data_cost, float *label_color, int nlabels, int nnodes, int nedges, int *edges,
           unsigned char *node_visible, float *sameview_prior, float smooth_prior, int *labels) {
  
  gl_p = new Params();
  Params &p = *gl_p;

  p.label_color = label_color;
  p.node_visible = node_visible;
  p.nnodes = nnodes;
  p.nlabels = nlabels;
  p.label_color = label_color;
  p.sameview_prior = sameview_prior;
  p.smooth_prior = smooth_prior;

  cerr << "Making gc" << endl;
  GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(p.nnodes, p.nlabels);
  gc->setVerbosity(1);

  cerr << "Adding edges" << endl;
  for (int ei = 0; ei < nedges; ei++) {
    gc->setNeighbors(edges[2*ei], edges[1+2*ei]);
    gc->setNeighbors(edges[2*ei+1], edges[2*ei]);
  }

  cerr << "nnodes = " << nnodes << " nlabels = " << nlabels << endl;

  cerr << "Adding costs" << endl;

  gc->setDataCost(data_cost);
  gc->setSmoothCost(smooth_cost);
  cerr << "Expansion" << endl;
  gc->expansion(GC_ITERS);

  cerr << "Assigning to labels" << endl;
  for (int i = 0; i < nnodes; i++) {
    labels[i] = gc->whatLabel(i);
  }

  cerr << "Done" << endl;
  delete gc;
  delete gl_p;
}

