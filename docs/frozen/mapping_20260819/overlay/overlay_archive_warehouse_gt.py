#!/usr/bin/env python3
"""Overlay reproduced archive maps/routes on warehouse GT in the up-LiDAR frame."""
from pathlib import Path
import hashlib, json, math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from flifltbot.mapping.factor_diag import load_odom_gt, apply_se2, umeyama_se2, sync_est_to_gt
from flifltbot.common.runtime.accuracy_eval import gt_segments, dist_seg

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'maps/20260815/sim/run_004137'
SRC = ROOT / 'results/20260819/archive_all_methods_reproduced/warehouse'
OUT = SRC / 'gt_lidar_corner'
LIDAR_X = 1.14
RECORDED_LIDAR_X = 0.95
GT_CORNERS = np.array([[-15., -25.], [15., -25.], [15., 25.], [-15., 25.]])
OUTER = gt_segments()[:4]

def lidar_xy(poses):
    a = np.asarray([[p['pose'][0], p['pose'][1], p['pose'][2]] for p in poses], float)
    if not len(a): return np.empty((0,2))
    return np.column_stack((a[:,0] + LIDAR_X*np.cos(a[:,2]), a[:,1] + LIDAR_X*np.sin(a[:,2])))

def bbox_corners(points, angle):
    c,s=math.cos(angle),math.sin(angle)
    q=np.column_stack((c*points[:,0]+s*points[:,1], -s*points[:,0]+c*points[:,1]))
    lo=np.quantile(q,[.01,.01],axis=0); hi=np.quantile(q,[.99,.99],axis=0)
    cs=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
    return apply_se2(cs, angle, 0, 0), cs

def fit(points):
    pts=np.asarray(points,float)
    if len(pts)>6000:
        pts=pts[np.linspace(0,len(pts)-1,6000).astype(int)]
    best=None
    for ang in np.linspace(-math.pi/2, math.pi/2, 361):
        rc, src_c = bbox_corners(pts, ang)
        yaw,tx,ty=umeyama_se2(rc, GT_CORNERS)
        tr=apply_se2(pts,yaw,tx,ty)
        # favor candidates whose long boundaries land on the four GT walls
        bd=np.array([min(abs(p[0]+15),abs(p[0]-15),abs(p[1]+25),abs(p[1]-25)) for p in tr])
        outer=np.minimum.reduce([np.abs(tr[:,0]+15),np.abs(tr[:,0]-15),np.abs(tr[:,1]+25),np.abs(tr[:,1]-25)]) < 2.0
        wall=float(np.median(bd[outer])) if np.count_nonzero(outer)>30 else 10.
        span=np.ptp(tr,axis=0)
        shape=float(np.linalg.norm(span-[30.,50.]))
        corner=float(np.sqrt(np.mean(np.sum((apply_se2(rc,yaw,tx,ty)-GT_CORNERS)**2,axis=1))))
        score=wall+0.15*corner+0.03*shape
        if best is None or score<best[0]: best=(score,yaw,tx,ty,corner,span)
    return best

def route_error(pred_poses, pred_xy, gt):
    if not len(pred_poses) or not len(gt): return None
    tt=np.asarray([g['t'] for g in gt]); gg=lidar_xy(gt)
    out=[]
    for p,xy in zip(pred_poses,pred_xy):
        t=float(p.get('t',0)); j=int(np.argmin(np.abs(tt-t)))
        if abs(tt[j]-t)<=.3: out.append((xy, gg[j]))
    if not out:return None
    a=np.asarray([x[0] for x in out]); b=np.asarray([x[1] for x in out]); e=np.linalg.norm(a-b,axis=1)
    return {'n':int(len(e)),'median_cm':float(np.median(e)*100),'rmse_cm':float(np.sqrt(np.mean(e*e))*100)}

def refine_outer_walls(points_world):
    """Small rigid correction using points already associated with GT outer walls."""
    p=np.asarray(points_world,float)
    targets=np.array([-15.,15.,-25.,25.])
    distances=np.column_stack((np.abs(p[:,0]+15),np.abs(p[:,0]-15),np.abs(p[:,1]+25),np.abs(p[:,1]-25)))
    assignment=np.argmin(distances,axis=1)
    mask=np.min(distances,axis=1)<1.25
    q=p[mask]; a=assignment[mask]
    best=None
    for delta in np.deg2rad(np.linspace(-3.,3.,241)):
        c,s=np.cos(delta),np.sin(delta)
        r=np.column_stack((c*q[:,0]-s*q[:,1],s*q[:,0]+c*q[:,1]))
        dx_terms=[]; dy_terms=[]
        for k in (0,1): dx_terms.extend((targets[k]-r[a==k,0]).tolist())
        for k in (2,3): dy_terms.extend((targets[k]-r[a==k,1]).tolist())
        dx=float(np.median(dx_terms)); dy=float(np.median(dy_terms))
        residual=[]
        for k in (0,1): residual.extend(np.abs(r[a==k,0]+dx-targets[k]).tolist())
        for k in (2,3): residual.extend(np.abs(r[a==k,1]+dy-targets[k]).tolist())
        score=float(np.median(residual))+0.15*float(np.percentile(residual,80))
        if best is None or score<best[0]: best=(score,delta,dx,dy)
    return best

def draw(ax, mp, gt_route, pred_lidar, method, row):
    for a,b in OUTER: ax.plot([a[0],b[0]],[a[1],b[1]],color='k',lw=1.5)
    for a,b in gt_segments()[4:]: ax.plot([a[0],b[0]],[a[1],b[1]],color='0.55',lw=.35,alpha=.6)
    ax.scatter(mp[:,0],mp[:,1],s=.18,c='#2474b5',alpha=.5,label='map')
    ax.plot(gt_route[:,0],gt_route[:,1],color='#18a558',lw=1,label='GT LiDAR route')
    ax.plot(pred_lidar[:,0],pred_lidar[:,1],color='#d62728',lw=1,label='pred LiDAR route')
    ax.scatter(GT_CORNERS[:,0],GT_CORNERS[:,1],c='orange',s=18,zorder=5)
    ax.set_aspect('equal'); ax.set_xlim(-18,18); ax.set_ylim(-28,28); ax.grid(alpha=.2)
    ax.set_title(f"{method} | route med {row['route_median_cm']:.1f} cm, RMSE {row['route_rmse_cm']:.1f} cm\nSE(2): yaw {row['yaw_deg']:.2f}°, t=({row['tx_m']:.2f},{row['ty_m']:.2f})")
    ax.legend(loc='upper right',fontsize=7)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gt=load_odom_gt(RUN)
    gt_route=lidar_xy(gt)
    rows=[]; rendered=[]
    files=sorted(SRC.glob('warehouse_*.json'))
    for path in files:
        d=json.loads(path.read_text()); pts=np.asarray(d['map'],float)
        est_xy, gt_xy, est_yaw, gt_yaw, _ = sync_est_to_gt(d.get('poses',[]), gt)
        gt_recorded_lidar=np.column_stack((gt_xy[:,0]+RECORDED_LIDAR_X*np.cos(gt_yaw),gt_xy[:,1]+RECORDED_LIDAR_X*np.sin(gt_yaw)))
        yaw,tx,ty=umeyama_se2(est_xy,gt_recorded_lidar)
        trajectory_yaw=yaw
        corner=0.0
        prelim=apply_se2(pts,yaw,tx,ty)
        _,delta,dx,dy=refine_outer_walls(prelim)
        c,s=math.cos(delta),math.sin(delta)
        tx,ty=c*tx-s*ty+dx,s*tx+c*ty+dy
        yaw += delta
        span=np.ptp(apply_se2(pts,yaw,tx,ty),axis=0)
        mp=apply_se2(pts,yaw,tx,ty)
        body=np.asarray([p['pose'] for p in d.get('poses',[])],float)
        pred_recorded_lidar=body[:,:2] if len(body) else np.empty((0,2))
        lever_delta=LIDAR_X-RECORDED_LIDAR_X
        pred_lidar=np.column_stack((pred_recorded_lidar[:,0]+lever_delta*np.cos(body[:,2]),pred_recorded_lidar[:,1]+lever_delta*np.sin(body[:,2]))) if len(body) else pred_recorded_lidar
        pred_lidar=apply_se2(pred_lidar,yaw,tx,ty)
        pred_recorded_world=apply_se2(pred_recorded_lidar,yaw,tx,ty)
        pred_lever=np.linalg.norm(pred_lidar-pred_recorded_world,axis=1)
        rr=route_error(d.get('poses',[]),pred_lidar,gt)
        route_hash=hashlib.sha256(np.ascontiguousarray(pred_lidar,dtype=np.float64).tobytes()).hexdigest()
        row={'method':d['method'],'yaw_deg':math.degrees(yaw),'trajectory_fit_yaw_deg':math.degrees(trajectory_yaw),'wall_refinement_yaw_deg':math.degrees(delta),'tx_m':tx,'ty_m':ty,'map_span_m':span.tolist(),'route_reference':'up LiDAR center at x=1.14 m','predicted_pose_semantics':'recorded horizon LiDAR pose at x=0.95 m','predicted_route_sha256':route_hash,'predicted_lidar_reprojection_delta_m':{'expected':lever_delta,'min':float(pred_lever.min()),'median':float(np.median(pred_lever)),'max':float(pred_lever.max()),'max_abs_error':float(np.max(np.abs(pred_lever-lever_delta)))},'route_n':rr['n'] if rr else 0,'route_median_cm':rr['median_cm'] if rr else None,'route_rmse_cm':rr['rmse_cm'] if rr else None}
        rows.append(row)
        np.savez_compressed(OUT/(path.stem+'_aligned_data.npz'),map=mp,gt_lidar_route=gt_route,pred_lidar_route=pred_lidar)
        one,one_ax=plt.subplots(figsize=(10,14))
        draw(one_ax,mp,gt_route,pred_lidar,d['method'],row)
        one.tight_layout(); one_path=OUT/(path.stem+'_vs_gt_lidar_corner.png'); one.savefig(one_path,dpi=180); plt.close(one)
        rendered.append(one_path)
    # The comparison is a literal collage of the individual PNGs. This avoids
    # renderer-dependent path simplification making a route look different.
    images=[Image.open(p).convert('RGB') for p in rendered]
    w=min(im.width for im in images); resized=[]
    for im in images:
        h=round(im.height*w/im.width); resized.append(im.resize((w,h),Image.LANCZOS))
    h=max(im.height for im in resized); canvas=Image.new('RGB',(2*w,2*h),'white')
    for i,im in enumerate(resized): canvas.paste(im,((i%2)*w,(i//2)*h))
    canvas.save(OUT/'warehouse_archive_methods_vs_gt_lidar_corner.png')
    gt_body=np.asarray([g['pose'][:2] for g in gt],float)
    gt_lever=np.linalg.norm(gt_route-gt_body,axis=1)
    (OUT/'summary.json').write_text(json.dumps({'scene':'warehouse','run':str(RUN),'route_reference':'up LiDAR center at x=1.14 m for both GT and prediction','lidar_x_m':LIDAR_X,'historical_run_model_lidar_x_m':RECORDED_LIDAR_X,'predicted_pose_semantics':'archive horizon NPZ pose is already the historical LiDAR pose; reproject by +0.19 m, never +1.14 m','alignment':'fit predicted historical LiDAR trajectory to GT historical LiDAR trajectory, then small rigid outer-wall refinement; same final transform for map and predicted route','gt_lidar_lever_m':{'expected':LIDAR_X,'min':float(gt_lever.min()),'median':float(np.median(gt_lever)),'max':float(gt_lever.max()),'max_abs_error':float(np.max(np.abs(gt_lever-LIDAR_X)))},'methods':rows},indent=2),encoding='utf-8')

if __name__=='__main__': main()
