#!/usr/bin/env python3
"""Re-run wall diagnostics under segment-window perturbations.

This is an offline audit helper; it never changes mapping/marker production
files. It writes a machine-readable report next to the requested analysis
directory.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("wall_geometry", HERE / "analyze_result_ply_wall_geometry.py")
M = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(M)

def run_variant(ply: Path, markers: Path, gt: Path, out: Path):
    allp = M.read_ascii_ply(ply); mpcloud, pmarks = M.marker_cloud(allp)
    md = json.loads(markers.read_text()); m98, corr = M.recover_marker_correspondence(pmarks, md, 98)
    nn = cKDTree(mpcloud[:, :2]).query(mpcloud[:, :2], k=2)[0][:, 1]
    spacing = float(np.median(nn)); seed=max(.01,.5*spacing); inlier=max(.008,.4*spacing)
    theta=M.estimate_angle(mpcloud[:,:2],.12,.14,seed,inlier); xy=M.rotate_xy(mpcloud[:,:2],theta); p=M.rotate_xy(m98[None,:2],theta)[0]
    def fit(cq,sq):
        w={n:M.fit_wall(xy,a,s,cq,sq,seed,inlier) for n,a,s in [('left',0,'low'),('right',0,'high'),('bottom',1,'low'),('top',1,'high')]}
        if any(v['inner'] is None for v in w.values()): return None
        o=np.array([min(w['left']['inner']['offset_center'],w['right']['inner']['offset_center']),min(w['bottom']['inner']['offset_center'],w['top']['inner']['offset_center'])])
        size=[abs(w['right']['inner']['offset_center']-w['left']['inner']['offset_center']),abs(w['top']['inner']['offset_center']-w['bottom']['inner']['offset_center'])]
        return {'size_m':size,'marker98_native_m':(p-o).tolist(),'inner_distances_m':{n:M.line_point_distance(p,v['inner'],v['axis']) for n,v in w.items()},'outer_available':{n:bool(v.get('outer')) for n,v in w.items()},'line_quality':{n:{k:(v[k] and {q:v[k][q] for q in ('support','coverage','residual_median_m','residual_p95_m','confidence') if q in v[k]}) for k in ('inner','outer')} for n,v in w.items()}}
    sensitivities=[]
    for cq in (.08,.12,.16,.20):
        for sq in (.30,.35,.40):
            x=fit(cq,sq)
            if x: sensitivities.append({'central_quantile':cq,'wall_side_quantile':sq,**x})
    label98=next(x for x in md['markers'] if int(x['marker_id'])==98)
    return {'input':{'path':str(ply),'sha256':hashlib.sha256(ply.read_bytes()).hexdigest(),'point_count':len(allp)},'marker_correspondence':corr,'point_spacing_m':spacing,'angle_deg':float(np.degrees(theta)),'baseline':fit(.12,.35),'sensitivity':sensitivities,'marker98_label_metrics':{k:label98[k] for k in ('point_xyz_m','observations','inliers','spread_m','spread_p95_m','ray_residual_m','ray_residual_p95_m')}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ply',type=Path,required=True);ap.add_argument('--marker-points',type=Path,required=True);ap.add_argument('--gt-floorplan',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True); report={'schema':'fliftbot.result_ply_wall_geometry_audit.v1','gt':json.loads(args.gt_floorplan.read_text()),'variant':run_variant(args.ply,args.marker_points,args.gt_floorplan,args.out_dir)}
    (args.out_dir/'wall_geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
