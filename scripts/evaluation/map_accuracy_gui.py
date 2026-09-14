#!/usr/bin/env python3
"""Interactive two-map alignment and accuracy checker.

Usage:
  python3 map_accuracy_gui.py --map MAP.png --gt GT.png

The first click selects a point on the left image, the second on the right.
Three or more correspondences enable a robust rigid (SE(2), no scale) fit.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

BG = '#919191'
PANEL = '#dddddd'
INK = '#222222'
GOOD = '#ACE1AF'
BAD = '#f18585'
ACCENTS = ['#d1495b', '#00798c', '#edae49', '#6a4c93', '#2a9d8f', '#e76f51', '#457b9d', '#8a5a44']
FONT = ('Lato', 17)
FONT_BOLD = ('Lato', 17, 'bold')
FONT_SMALL = ('Lato', 15)
POINT_RADIUS = 8


def fit_transform(src, dst, weights=None, allow_scale=True):
    """Return similarity A,t,scale mapping src to dst."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = np.maximum(w, 1e-9); w /= w.sum()
    a = (src * w[:, None]).sum(0); b = (dst * w[:, None]).sum(0)
    cov=((src-a)*w[:, None]).T @ (dst-b)
    u, singular, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0: vt[-1] *= -1; r = vt.T @ u.T
    variance=float(np.sum(w[:,None]*(src-a)**2))
    scale=float(singular.sum()/variance) if allow_scale and variance>1e-12 else 1.0
    transform=scale*r
    return transform, b-transform@a, scale


def robust_fit(src, dst, allow_scale=True, tolerance_px=12.0):
    """Deterministic RANSAC followed by robust refinement on inliers."""
    src,dst=np.asarray(src,float),np.asarray(dst,float); n=len(src)
    total=math.comb(n,3)
    if total<=12000:
        candidates=list(itertools.combinations(range(n),3))
    else:
        rng=np.random.default_rng(20260905); chosen=set()
        while len(chosen)<4000:
            chosen.add(tuple(sorted(rng.choice(n,3,replace=False).tolist())))
        candidates=sorted(chosen)
    best=None
    for ids in candidates:
        ids=np.asarray(ids); a,b=src[ids],dst[ids]
        area_a=abs((a[1,0]-a[0,0])*(a[2,1]-a[0,1])-(a[1,1]-a[0,1])*(a[2,0]-a[0,0]))
        area_b=abs((b[1,0]-b[0,0])*(b[2,1]-b[0,1])-(b[1,1]-b[0,1])*(b[2,0]-b[0,0]))
        if area_a<4 or area_b<4: continue
        A,t,s=fit_transform(a,b,allow_scale=allow_scale); err=np.linalg.norm(src@A.T+t-dst,axis=1); inside=err<=tolerance_px
        if inside.sum()<3: continue
        score=(int(inside.sum()),-float(np.median(err[inside])))
        if best is None or score>best[0]: best=(score,inside,A,t,s)
    if best is None: raise ValueError('Выберите точки, охватывающие площадь, а не одну линию.')
    _,inside,A,t,s=best
    for _ in range(20):
        err=np.linalg.norm(src[inside]@A.T+t-dst[inside],axis=1); mad=max(1.4826*float(np.median(np.abs(err-np.median(err)))),1.0); weights=1/np.maximum(1,err/(2.5*mad)); nA,nt,ns=fit_transform(src[inside],dst[inside],weights,allow_scale)
        if np.max(np.abs(nA-A))<1e-9 and np.max(np.abs(nt-t))<1e-7: break
        A,t,s=nA,nt,ns
    err=np.linalg.norm(src@A.T+t-dst,axis=1); inside=err<=tolerance_px
    # Small deterministic coordinate search targets the median residual, which
    # is less sensitive to a user's few-pixel click error than plain LS.
    theta=math.atan2(A[1,0],A[0,0]); log_s=math.log(max(s,1e-12)); params=np.array([theta,log_s,t[0],t[1]])
    def model(v):
        sc=math.exp(v[1]) if allow_scale else 1.0; co,si=math.cos(v[0]),math.sin(v[0]); return sc*np.array([[co,-si],[si,co]]),v[2:]
    def objective(v):
        M,mt=model(v); e=np.linalg.norm(src[inside]@M.T+mt-dst[inside],axis=1); trimmed=np.sort(e)[:max(3,int(math.ceil(.8*len(e))))]; return float(np.median(e)+.15*np.mean(trimmed))
    steps=np.array([math.radians(.35),.006 if allow_scale else 0.0,1.5,1.5]); value=objective(params)
    for _ in range(9):
        improved=True
        while improved:
            improved=False
            for j in range(4):
                if steps[j]==0: continue
                for sign in (-1,1):
                    trial=params.copy();trial[j]+=sign*steps[j];score=objective(trial)
                    if score+1e-9<value:params,value=trial,score;improved=True
        steps*=.5
    A,t=model(params);s=float(math.sqrt(max(np.linalg.det(A),0.0)));err=np.linalg.norm(src@A.T+t-dst,axis=1);inside=err<=tolerance_px
    return A,t,inside,s


def object_mask(image):
    """Foreground mask for transparent drawings and black-on-light maps."""
    a=np.asarray(image.convert('RGBA')); alpha=a[:,:,3]
    if np.any(alpha<250): return alpha>20
    gray=a[:,:,:3].mean(axis=2)
    if np.percentile(gray,90)>180: return gray<120
    corners=np.vstack((a[:10,:10,:3].reshape(-1,3),a[:10,-10:,:3].reshape(-1,3),a[-10:,:10,:3].reshape(-1,3),a[-10:,-10:,:3].reshape(-1,3))); bg=np.median(corners,axis=0)
    return np.linalg.norm(a[:,:,:3]-bg,axis=2)>20


class MapAccuracyGUI(tk.Tk):
    def __init__(self, map_path, gt_path):
        super().__init__()
        self.title('Map accuracy — point correspondence')
        # Give the controls and the two image panes enough room on a normal
        # desktop.  The minimum is deliberately generous: tiny Tk windows
        # make the point labels and file names hard to read.
        self.geometry('1680x1080'); self.minsize(1200, 820)
        self.configure(background=BG)
        self.map_path, self.gt_path = Path(map_path), Path(gt_path)
        self.images = [Image.open(self.map_path).convert('RGBA'), Image.open(self.gt_path).convert('RGBA')]
        self.names = [self.map_path.name, self.gt_path.name]
        self.points = [[], []]; self.links = []; self.pending = None
        self.zoom = [1.0, 1.0]; self.pan = [np.zeros(2), np.zeros(2)]
        self.tk_images = [None, None]
        # A resized image is cached independently from the clipped viewport.
        # Panning therefore only crops an existing bitmap and never resizes it
        # or creates a large collection of Canvas items on every mouse event.
        self._scaled_cache = [None, None]
        self.canvases = []; self.hover = None; self._drag = None
        self._redraw_pending = False
        self.show_links_var = tk.BooleanVar(value=True)
        self.allow_scale_var = tk.BooleanVar(value=True)
        self.edit_index = None
        self._build_ui(); self._show_points_view()
        self.bind('<Escape>', self._cancel_pending)

    def _build_ui(self):
        style=ttk.Style(self); style.theme_use('clam')
        available_fonts = set(tkfont.families(self))
        family = next((name for name in ('Inter', 'Segoe UI', 'Noto Sans', 'Lato', 'DejaVu Sans') if name in available_fonts), 'TkDefaultFont')
        self.ui_font = (family, 17)
        self.ui_font_bold = (family, 17, 'bold')
        self.dialog_font = (family, 18)
        self.dialog_font_bold = (family, 18, 'bold')
        style.configure('TFrame', background=BG)
        style.configure('Panel.TFrame', background=PANEL)
        style.configure('TLabel', background=BG, foreground=INK, font=self.ui_font)
        style.configure('Title.TLabel', background=BG, foreground=INK, font=(family, 24, 'bold'))
        style.configure('TButton', font=self.ui_font, padding=(14, 10), relief='flat')
        style.map('TButton', background=[('active', '#a9a9a9')])
        style.configure('TCheckbutton', background=BG, foreground=INK, font=self.ui_font)
        top = ttk.Frame(self, padding=10); top.pack(fill='x')
        ttk.Label(top, text='Интерактивная оценка карты', style='Title.TLabel').pack(side='left')
        self.status = ttk.Label(top, text='Связей: 0 | Нужно минимум 3'); self.status.pack(side='right')
        files=ttk.Frame(self,padding=(10,0,10,8));files.pack(fill='x')
        self.file_labels=[]
        for idx,title in enumerate(('Карта','GT-разметка')):
            ttk.Label(files,text=title+':').pack(side='left',padx=(0,4)); lab=ttk.Label(files,text=self.names[idx]);lab.pack(side='left',padx=(0,6));self.file_labels.append(lab);ttk.Button(files,text='Выбрать…',command=lambda i=idx:self._choose_file(i)).pack(side='left',padx=(0,18))
        self.body = ttk.Frame(self); self.body.pack(fill='both', expand=True, padx=8)
        self.stage = tk.Canvas(self.body, background='#30343b', highlightthickness=1, highlightbackground='#888')
        self.stage.pack(fill='both', expand=True)
        # Both logical image panes share one Canvas.  This keeps point-to-point
        # links continuous and gives pan/zoom a single coordinate system.
        self.canvases = [self.stage, self.stage]
        # Ctrl+left-click places a correspondence point. A plain left drag
        # pans only the pane below the cursor; a plain click does nothing.
        self.stage.bind('<Button-1>', self._stage_left_press)
        self.stage.bind('<B1-Motion>', self._stage_left_motion)
        self.stage.bind('<ButtonRelease-1>', self._stage_left_release)
        self.stage.bind('<Motion>', self._stage_motion)
        self.stage.bind('<Leave>', self._stage_leave)
        self.stage.bind('<Button-2>', self._stage_begin_pan)
        self.stage.bind('<B2-Motion>', self._stage_pan_motion)
        self.stage.bind('<ButtonRelease-2>', self._stage_end_pan)
        self.stage.bind('<MouseWheel>', self._stage_wheel_zoom)
        self.stage.bind('<Button-4>', lambda e: self._stage_wheel_zoom(e, 1))
        self.stage.bind('<Button-5>', lambda e: self._stage_wheel_zoom(e, -1))
        self.stage.bind('<Configure>', lambda _e: self._request_redraw())
        # Keep the toolbar on two comfortable rows.  With a larger typeface a
        # single row makes the right-most actions disappear on 1280px screens.
        bottom = ttk.Frame(self, padding=(10, 8, 10, 6)); bottom.pack(fill='x')
        zoom_row = ttk.Frame(bottom); zoom_row.pack(fill='x', pady=(0, 5))
        ttk.Label(zoom_row, text='Масштаб:', font=self.ui_font_bold).pack(side='left', padx=(0, 8))
        ttk.Button(zoom_row, text='По размеру', command=self._fit_all).pack(side='left')
        ttk.Button(zoom_row, text='−', width=3, command=lambda: self._zoom_all(0.85)).pack(side='left', padx=(5, 0))
        ttk.Button(zoom_row, text='+', width=3, command=lambda: self._zoom_all(1.18)).pack(side='left', padx=(3, 16))
        ttk.Checkbutton(zoom_row, text='Показывать связи', variable=self.show_links_var, command=self._redraw_all).pack(side='left')
        ttk.Checkbutton(zoom_row, text='Учитывать масштаб', variable=self.allow_scale_var).pack(side='left', padx=(14, 4))
        action_row = ttk.Frame(bottom); action_row.pack(fill='x')
        ttk.Button(action_row, text='Изменить выбранную', command=self._edit_selected).pack(side='left')
        ttk.Button(action_row, text='Удалить выбранную связь', command=self._delete_selected).pack(side='left', padx=(6, 0))
        ttk.Button(action_row, text='Удалить все', command=self._clear).pack(side='left', padx=(6, 0))
        ttk.Button(action_row, text='Сохранить связи', command=self._save).pack(side='right')
        ttk.Button(action_row, text='Оценить наложение', command=self._evaluate).pack(side='right', padx=(0, 6))
        self.listbox = tk.Listbox(self, height=5, exportselection=False, font=self.ui_font, bg='#f6f1e8', fg=INK, selectbackground='#7b9e9e', activestyle='none')
        self.listbox.pack(fill='x', padx=8)
        self.listbox.bind('<Delete>', lambda e: self._delete_selected())
        ttk.Label(self, text='Ctrl+ЛКМ — поставить точку   •   ЛКМ + перетаскивание — двигать панель   •   Колесо — масштаб   •   Esc — отмена', foreground='#4d4d4d', background=BG, font=(self.ui_font[0], 15)).pack(fill='x', padx=10, pady=(0, 9))

    def _stage_pane(self, i):
        gap = 120
        width = max(1, self.stage.winfo_width())
        height = max(1, self.stage.winfo_height())
        pane_width = max(1, (width - gap) / 2)
        return (0 if i == 0 else pane_width + gap, 0, pane_width, height)

    def _stage_pane_at(self, x, y):
        for i in (0, 1):
            px, py, pw, ph = self._stage_pane(i)
            if px <= x <= px + pw and py <= y <= py + ph:
                return i
        return None

    def _stage_click(self, event):
        """Compatibility wrapper; point placement is now Ctrl+left-click."""
        i = self._stage_pane_at(event.x, event.y)
        if i is not None and (event.state & 0x0004):
            self._click(i, event)

    def _stage_left_press(self, event):
        i = self._stage_pane_at(event.x, event.y)
        if i is None:
            return
        if event.state & 0x0004:  # Control on X11/Windows Tk
            self._click(i, event)
            self._drag = None
            return
        self._begin_pan(i, event)

    def _stage_left_motion(self, event):
        if self._drag is not None:
            self._pan_motion(self._drag[0], event)
            return
        self._stage_motion(event)

    def _stage_left_release(self, event):
        if self._drag is not None:
            self._end_pan(self._drag[0], event)

    def _stage_motion(self, event):
        i = self._stage_pane_at(event.x, event.y)
        if i is not None:
            self._motion(i, event)
        elif self.pending is not None:
            self.hover = None
            self._request_redraw()

    def _stage_leave(self, _event):
        self._leave(0); self._leave(1)

    def _stage_begin_pan(self, event):
        i = self._stage_pane_at(event.x, event.y)
        if i is not None:
            self._begin_pan(i, event)

    def _stage_pan_motion(self, event):
        if self._drag is not None:
            self._pan_motion(self._drag[0], event)

    def _stage_end_pan(self, event):
        if self._drag is not None:
            self._end_pan(self._drag[0], event)

    def _stage_wheel_zoom(self, event, direction=None):
        i = self._stage_pane_at(event.x, event.y)
        if i is not None:
            self._wheel_zoom(i, event, direction)

    def _fit_scale(self, i):
        _x, _y, pane_width, pane_height = self._stage_pane(i)
        iw, ih = self.images[i].size
        return min(max((pane_width - 8) / iw, .01), max((pane_height - 8) / ih, .01))

    def _view(self, i):
        """Return display scale and image origin in canvas coordinates."""
        s = self._fit_scale(i) * self.zoom[i]
        iw, ih = self.images[i].size
        pane_x, pane_y, pane_width, pane_height = self._stage_pane(i)
        origin = np.array([pane_x + (pane_width - iw * s) / 2,
                           pane_y + (pane_height - ih * s) / 2]) + self.pan[i]
        return s, origin

    def _xy(self, i, event):
        s, origin = self._view(i)
        p = (np.array([event.x, event.y], float) - origin) / s
        iw, ih = self.images[i].size
        if p[0] < 0 or p[1] < 0 or p[0] >= iw or p[1] >= ih:
            return None
        return p

    def _click(self, i, event):
        p=self._xy(i,event)
        if p is None:
            return
        if i == 0:
            self.pending = p; self.hover = None
            if self.edit_index is None:
                self.status.configure(text='Точка карты выбрана — наведите курсор на GT и кликните соответствующую точку')
            else:
                self.status.configure(text='Точка карты обновлена — наведите курсор на GT и кликните соответствующую точку')
        elif self.pending is not None:
            if self.edit_index is None:
                self.points[0].append(self.pending); self.points[1].append(p)
            else:
                self.points[0].insert(self.edit_index, self.pending); self.points[1].insert(self.edit_index, p)
            self.links = list(range(1, len(self.points[0]) + 1))
            self.pending=None; self.edit_index=None; self.hover=None
            self._edit_backup = None
            self._invalidate_result()
            self._refresh_list(); self._redraw(0); self._redraw(1); self._set_status()

    def _motion(self, i, event):
        if self.pending is not None:
            p=self._xy(i,event)
            self.hover = None if p is None else (i,p)
            self._request_redraw(0)

    def _leave(self, i):
        if self.pending is not None and self.hover is not None and self.hover[0] == i:
            self.hover = None
            self._redraw_all()

    def _cancel_pending(self, _event=None):
        """Escape cancels an unfinished pair (and restores an edited pair)."""
        if self.pending is None and self.edit_index is None:
            return 'break'
        if self.edit_index is not None and getattr(self, '_edit_backup', None) is not None:
            index, map_point, gt_point = self._edit_backup
            self.points[0].insert(index, map_point)
            self.points[1].insert(index, gt_point)
        self.pending = None
        self.hover = None
        self.edit_index = None
        self._edit_backup = None
        self._refresh_list()
        self._redraw_all()
        self._set_status()
        return 'break'

    def _begin_pan(self, i, event):
        self._drag = (i, event.x, event.y, self.pan[i].copy())
        self.canvases[i].configure(cursor='fleur')

    def _pan_motion(self, i, event):
        if self._drag is None or self._drag[0] != i:
            return
        _, x0, y0, p0 = self._drag
        self.pan[i] = p0 + np.array([event.x - x0, event.y - y0], float)
        self._request_redraw(i)

    def _end_pan(self, i, _event):
        self._drag = None
        self.canvases[i].configure(cursor='')

    def _wheel_zoom(self, i, event, direction=None):
        """Zoom around cursor without moving the image point under it."""
        direction = direction if direction is not None else (1 if event.delta > 0 else -1)
        old_s, old_origin = self._view(i)
        image_point = (np.array([event.x, event.y], float) - old_origin) / old_s
        self.zoom[i] = float(np.clip(self.zoom[i] * (1.15 if direction > 0 else 1 / 1.15), .25, 8.0))
        new_s = self._fit_scale(i) * self.zoom[i]
        iw, ih = self.images[i].size
        pane_x, pane_y, pane_width, pane_height = self._stage_pane(i)
        centered = np.array([pane_x + (pane_width - iw * new_s) / 2,
                             pane_y + (pane_height - ih * new_s) / 2])
        self.pan[i] = np.array([event.x, event.y]) - image_point * new_s - centered
        self._request_redraw()

    def _fit_all(self):
        self.zoom = [1.0, 1.0]; self.pan = [np.zeros(2), np.zeros(2)]
        self._redraw_all()

    def _zoom_all(self, factor):
        self.zoom = [float(np.clip(z * factor, .25, 8.0)) for z in self.zoom]
        self._redraw_all()

    def _redraw_all(self):
        self._request_redraw()

    def _request_redraw(self, _i=None):
        """Coalesce configure/mouse redraws so rapid motion never queues work."""
        if self._redraw_pending:
            return
        self._redraw_pending = True
        self.after_idle(self._redraw_deferred)

    def _redraw_deferred(self):
        self._redraw_pending = False
        self._redraw()

    def _refresh_list(self):
        self.listbox.delete(0,'end')
        for n,(a,b) in enumerate(zip(self.points[0],self.points[1]),1): self.listbox.insert('end', f'{n}: map ({a[0]:.1f}, {a[1]:.1f})  ↔  GT ({b[0]:.1f}, {b[1]:.1f})')

    def _set_status(self): self.status.configure(text=f'Связей: {len(self.points[0])} | ' + ('Готово к оценке' if len(self.points[0])>=3 else 'Нужно минимум 3'))

    def _delete_selected(self):
        sel=self.listbox.curselection()
        if not sel: return
        j=sel[0]; del self.points[0][j]; del self.points[1][j]; self.links = list(range(1, len(self.points[0]) + 1)); self._invalidate_result(); self._refresh_list(); self._redraw_all(); self._set_status()

    def _edit_selected(self):
        sel=self.listbox.curselection()
        if not sel: return
        j=sel[0]; self.pending=self.points[0][j]
        self._edit_backup = (j, self.points[0][j].copy(), self.points[1][j].copy())
        del self.points[0][j]; del self.points[1][j]
        self.edit_index = j
        self.links = list(range(1, len(self.points[0]) + 1)); self._invalidate_result(); self._refresh_list(); self._redraw_all()
        self.status.configure(text='Связь редактируется — при необходимости выберите новую точку карты, затем кликните GT')

    def _clear(self):
        self.points=[[],[]]; self.links=[]; self.pending=None; self.hover=None; self.edit_index=None; self._edit_backup=None; self._invalidate_result(); self._refresh_list(); self._redraw_all(); self._set_status()

    def _invalidate_result(self):
        if hasattr(self, '_last'):
            del self._last

    def _redraw(self, _i=None):
        if not hasattr(self, 'stage'):
            return
        c = self.stage
        c.delete('all')
        width, height = c.winfo_width(), c.winfo_height()
        gap = 120
        c.create_rectangle(0, 0, width / 2 - gap / 2, height, fill='#30343b', outline='#777')
        c.create_rectangle(width / 2 + gap / 2, 0, width, height, fill='#30343b', outline='#777')
        c.create_rectangle(width / 2 - gap / 2, 0, width / 2 + gap / 2, height, fill=BG, outline='')
        for pane_i in (0, 1):
            s, origin = self._view(pane_i)
            pane_x, pane_y, pane_width, pane_height = self._stage_pane(pane_i)
            # Render a viewport-sized bitmap. This is the Canvas equivalent of
            # clipping to a pane and guarantees that a zoomed image can never
            # paint over the other image or the central gutter.
            viewport = self._pane_bitmap(pane_i, s, origin, pane_x, pane_y, pane_width, pane_height)
            self.tk_images[pane_i] = ImageTk.PhotoImage(viewport)
            c.create_image(int(pane_x), int(pane_y), anchor='nw', image=self.tk_images[pane_i])
            c.create_text(pane_x + 12, 12, text=self.names[pane_i], fill='white', anchor='nw', font=FONT_BOLD)
            c.create_rectangle(pane_x, pane_y, pane_x + pane_width, pane_y + pane_height, outline='#888')

        # One continuous line per correspondence. It is drawn in the common
        # Canvas coordinate system, so it cannot kink at the former widget gap.
        if self.show_links_var.get():
            for j, (a, b) in enumerate(zip(self.points[0], self.points[1]), 1):
                sa, oa = self._view(0); sb, ob = self._view(1)
                pa = oa + np.asarray(a) * sa; pb = ob + np.asarray(b) * sb
                lx,ly,lw,lh=self._stage_pane(0);rx,ry,rw,rh=self._stage_pane(1)
                if not (lx<=pa[0]<=lx+lw and ly<=pa[1]<=ly+lh and rx<=pb[0]<=rx+rw and ry<=pb[1]<=ry+rh):
                    continue
                col = ACCENTS[(j - 1) % len(ACCENTS)]
                c.create_line(pa[0], pa[1], pb[0], pb[1], fill=col, width=3, dash=(8, 4), capstyle='round')
                c.create_text((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2 - 10,
                              text=str(j), fill=col, font=FONT_BOLD)
            if self.pending is not None and self.hover is not None and self.hover[0] == 1:
                sa, oa = self._view(0); sb, ob = self._view(1)
                pa = oa + np.asarray(self.pending) * sa; pb = ob + self.hover[1] * sb
                c.create_line(pa[0], pa[1], pb[0], pb[1], fill='white', width=3,
                              dash=(8, 4), capstyle='round')

        for pane_i in (0, 1):
            s, origin = self._view(pane_i)
            pane_x,pane_y,pane_width,pane_height=self._stage_pane(pane_i)
            for j, p in enumerate(self.points[pane_i]):
                x, y = origin + np.asarray(p) * s; col = ACCENTS[j % len(ACCENTS)]
                if not (pane_x+POINT_RADIUS<=x<=pane_x+pane_width-POINT_RADIUS and pane_y+POINT_RADIUS<=y<=pane_y+pane_height-POINT_RADIUS):
                    continue
                c.create_oval(x - POINT_RADIUS, y - POINT_RADIUS, x + POINT_RADIUS, y + POINT_RADIUS,
                              fill=col, outline='white', width=2)
                c.create_text(x + 13, y - 11, text=str(j + 1), fill='white', anchor='sw', font=FONT_BOLD)

    def _pane_bitmap(self, i, scale, origin, pane_x, pane_y, pane_width, pane_height):
        """Return exactly one pane-sized image, with the source clipped."""
        pw, ph = max(1, int(pane_width)), max(1, int(pane_height))
        viewport = Image.new('RGBA', (pw, ph), '#30343b')
        iw,ih=self.images[i].size
        display_left=max(float(pane_x),float(origin[0])); display_top=max(float(pane_y),float(origin[1]))
        display_right=min(float(pane_x+pane_width),float(origin[0]+iw*scale)); display_bottom=min(float(pane_y+pane_height),float(origin[1]+ih*scale))
        if display_right>display_left and display_bottom>display_top:
            source_box=((display_left-origin[0])/scale,(display_top-origin[1])/scale,(display_right-origin[0])/scale,(display_bottom-origin[1])/scale)
            crop=self.images[i].crop(source_box)
            dw=max(1,int(round(display_right-display_left))); dh=max(1,int(round(display_bottom-display_top)))
            crop=crop.resize((dw,dh),Image.Resampling.BILINEAR)
            dx=int(round(display_left-pane_x));dy=int(round(display_top-pane_y));viewport.alpha_composite(crop,(dx,dy))
        return viewport

    def _redraw_connector(self):
        # Kept as a compatibility no-op for callers from older versions.
        return

    def _file_dialog(self, mode, title, initial_path=None, default_name='alignment.png', default_suffix='.png'):
        """Large, keyboard-friendly file dialog.

        Tk's platform file dialogs inherit a very small size on some Linux
        window managers and do not expose a reliable size option.  This
        lightweight ttk dialog keeps the familiar file-browser behavior while
        giving the user a readable 1050x700 view and a path field for direct
        navigation.  It is intentionally local to this tool, so no desktop
        integration or extra dependency is required.
        """
        initial = Path(initial_path) if initial_path else Path.cwd()
        if initial.is_file():
            initial = initial.parent
        if not initial.exists():
            initial = Path.cwd()
        current = {'path': initial.resolve()}
        allowed = {'.png', '.pgm', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        result = {'value': None}
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.geometry('1050x700')
        dialog.minsize(850, 560)
        dialog.configure(background=BG)
        dialog.transient(self)
        dialog.grab_set()

        header = ttk.Frame(dialog, padding=(18, 16, 18, 10))
        header.pack(fill='x')
        ttk.Label(header, text=title, font=self.dialog_font_bold).pack(anchor='w')
        path_var = tk.StringVar(value=str(current['path']))
        path_row = ttk.Frame(dialog, padding=(18, 0, 18, 12))
        path_row.pack(fill='x')
        path_entry = ttk.Entry(path_row, textvariable=path_var, font=self.dialog_font)
        path_entry.pack(side='left', fill='x', expand=True, ipady=5)

        body = ttk.Frame(dialog, padding=(18, 0, 18, 8))
        body.pack(fill='both', expand=True)
        list_frame = ttk.Frame(body)
        list_frame.pack(fill='both', expand=True)
        listing = tk.Listbox(list_frame, font=self.dialog_font, activestyle='none',
                             bg='#f6f1e8', fg=INK, selectbackground='#7b9e9e',
                             selectforeground='white', borderwidth=0,
                             highlightthickness=1, highlightcolor='#7b9e9e')
        scroll = ttk.Scrollbar(list_frame, orient='vertical', command=listing.yview)
        listing.configure(yscrollcommand=scroll.set)
        listing.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        filename_var = tk.StringVar()
        if mode == 'save':
            name_row = ttk.Frame(dialog, padding=(18, 0, 18, 8))
            name_row.pack(fill='x')
            ttk.Label(name_row, text='Имя файла:', font=self.dialog_font).pack(side='left', padx=(0, 10))
            ttk.Entry(name_row, textvariable=filename_var, font=self.dialog_font).pack(side='left', fill='x', expand=True, ipady=5)

        footer = ttk.Frame(dialog, padding=(18, 6, 18, 16))
        footer.pack(fill='x')
        hint = ttk.Label(footer, text='', font=FONT_SMALL, foreground='#4d4d4d')
        hint.pack(side='left', fill='x', expand=True)

        def refresh(path=None):
            if path is not None:
                candidate = Path(path).expanduser()
                if not candidate.is_absolute():
                    candidate = current['path'] / candidate
                candidate = candidate.resolve()
                if candidate.is_file():
                    candidate = candidate.parent
                if not candidate.is_dir():
                    hint.configure(text='Папка не найдена')
                    return
                current['path'] = candidate
            path_var.set(str(current['path']))
            listing.delete(0, 'end')
            try:
                dirs = sorted((p for p in current['path'].iterdir() if p.is_dir()), key=lambda p: p.name.lower())
                files = sorted((p for p in current['path'].iterdir() if p.is_file() and p.suffix.lower() in allowed), key=lambda p: p.name.lower())
            except OSError as exc:
                hint.configure(text=f'Не удалось прочитать папку: {exc}')
                return
            for p in dirs:
                listing.insert('end', f'[ПАПКА]  {p.name}')
            for p in files:
                listing.insert('end', p.name)
            hint.configure(text=f'{len(dirs)} папок, {len(files)} изображений')

        def navigate(_event=None):
            refresh(path_var.get())

        def up():
            refresh(current['path'].parent)

        def selected_path():
            sel = listing.curselection()
            if not sel:
                return None
            text = listing.get(sel[0])
            return current['path'] / text.replace('[ПАПКА]  ', '', 1)

        def activate(_event=None):
            chosen = selected_path()
            if chosen is None:
                return
            if chosen.is_dir():
                refresh(chosen)
            elif mode == 'open':
                result['value'] = str(chosen)
                dialog.destroy()
            else:
                filename_var.set(chosen.name)

        def accept():
            if mode == 'open':
                chosen = selected_path()
                if chosen is None or not chosen.is_file():
                    hint.configure(text='Выберите файл изображения')
                    return
                result['value'] = str(chosen)
            else:
                name = filename_var.get().strip()
                if not name:
                    hint.configure(text='Укажите имя файла')
                    return
                chosen = Path(name).expanduser()
                if not chosen.is_absolute():
                    chosen = current['path'] / chosen
                if not chosen.suffix:
                    chosen = chosen.with_suffix(default_suffix)
                result['value'] = str(chosen)
            dialog.destroy()

        ttk.Button(path_row, text='Перейти', command=navigate).pack(side='left', padx=(10, 0))
        ttk.Button(path_row, text='Вверх', command=up).pack(side='left', padx=(6, 0))
        listing.bind('<Double-1>', activate)
        listing.bind('<Return>', activate)
        path_entry.bind('<Return>', navigate)
        if mode == 'save':
            filename_var.set(default_name)
        ttk.Button(footer, text='Отмена', command=dialog.destroy).pack(side='right', padx=(8, 0))
        ttk.Button(footer, text='Выбрать' if mode == 'open' else 'Сохранить', command=accept).pack(side='right')
        refresh()
        path_entry.focus_set()
        self.wait_window(dialog)
        return result['value']

    def _choose_file(self, i):
        p=self._file_dialog('open', 'Выберите '+('карту' if i==0 else 'GT-разметку'), self.map_path if i==0 else self.gt_path)
        if not p:return
        try:
            if hasattr(self, 'overlay'):
                self._restore(self.overlay_bar)
            self.images[i]=Image.open(p).convert('RGBA'); self.names[i]=Path(p).name
            if i==0:self.map_path=Path(p)
            else:self.gt_path=Path(p)
            self.file_labels[i].configure(text=self.names[i])
            self._scaled_cache[i] = None; self._clear()
        except Exception as e:messagebox.showerror('Не удалось открыть файл',str(e))

    def _show_points_view(self):
        self._redraw()

    def _evaluate(self):
        if len(self.points[0])<3: messagebox.showwarning('Недостаточно точек','Нужно выбрать минимум 3 пары точек.'); return
        src,dst=np.array(self.points[0]),np.array(self.points[1])
        try: A,t,inliers,scale=robust_fit(src,dst,self.allow_scale_var.get())
        except ValueError as e: messagebox.showerror('Наложение невозможно',str(e)); return
        aligned=src@A.T+t; err=np.linalg.norm(aligned-dst,axis=1); used=err[inliers]; med=float(np.median(used)); p95=float(np.percentile(used,95)); ratio=float(inliers.mean())
        if inliers.sum()<3 or ratio<.6: verdict='Плохое соответствие: точки противоречат друг другу.'
        elif med>12: verdict='Наложение сомнительное: проверьте отмеченные точки.'
        else: verdict='Наложение устойчивое.'
        self._last={'n':len(err),'inliers':np.flatnonzero(inliers).tolist(),'outliers':np.flatnonzero(~inliers).tolist(),'inlier_ratio':ratio,'median_px':med,'p95_px':p95,'max_px':float(used.max()),'errors_px':err.tolist(),'rotation_deg':math.degrees(math.atan2(A[1,0],A[0,0])),'scale':scale,'translation_px':t.tolist(),'verdict':verdict}
        self._show_overlay(A,t,verdict+f'  Inliers {inliers.sum()}/{len(inliers)}, масштаб {scale:.5f}',med,p95)

    def _show_overlay(self,r,t,verdict,med,p95):
        self.stage.pack_forget(); self.listbox.pack_forget()
        self.overlay=tk.Canvas(self.body,background='#30343b'); self.overlay.pack(fill='both',expand=True); self.overlay.bind('<Configure>',lambda e:self._draw_overlay(r,t))
        self._overlay_transform=(r,t)
        bar=ttk.Frame(self,padding=8);bar.pack(fill='x'); good=med<=12; ttk.Label(bar,text=f'{verdict}  median={med:.2f}px  P95={p95:.2f}px',background=GOOD if good else BAD,foreground=INK,padding=7,font=('Lato',14,'bold')).pack(side='left');ttk.Button(bar,text='Сохранить PNG',command=self._save_overlay_png).pack(side='right',padx=5);ttk.Button(bar,text='Вернуться к точкам',command=lambda:self._restore(bar)).pack(side='right');self.overlay_bar=bar
        self._draw_overlay(r,t)

    def _save_overlay_png(self):
        if not hasattr(self,'_overlay_transform'): return
        p=self._file_dialog('save', 'Сохранить наложение и тепловую карту', Path.cwd())
        if not p: return
        r,t=self._overlay_transform; h,w=np.asarray(self.images[1]).shape[:2]
        gt_mask=object_mask(self.images[1]); src_mask=object_mask(self.images[0]); yy,xx=np.where(src_mask); xy=np.column_stack((xx,yy)).astype(float)@r.T+t; xi=np.rint(xy[:,0]).astype(int);yi=np.rint(xy[:,1]).astype(int); keep=(xi>=0)&(yi>=0)&(xi<w)&(yi<h)
        mapped=np.zeros((h,w),bool); mapped[yi[keep],xi[keep]]=True
        rgb=np.full((h,w,3),255,np.uint8); rgb[gt_mask]=[0,0,0]; rgb[mapped]=[220,40,40]
        result=Image.fromarray(rgb,'RGB').convert('RGBA')
        draw=ImageDraw.Draw(result)
        for j,(a,b) in enumerate(zip(self.points[0],self.points[1])):
            q=np.asarray(a)@r.T+t; col=ACCENTS[j%len(ACCENTS)]; point_rgb=tuple(int(col[k:k+2],16) for k in (1,3,5)); draw.line((q[0],q[1],b[0],b[1]),fill=point_rgb,width=2)
            draw.ellipse((b[0]-6,b[1]-6,b[0]+6,b[1]+6),fill=point_rgb,outline='white',width=2)
        result.save(p)
        # True geometry heatmap: distance of every mapped contour pixel to GT.
        try:
            from scipy.ndimage import distance_transform_edt
            distance=distance_transform_edt(~gt_mask); vals=distance[yi[keep],xi[keep]]; ceiling=max(float(np.percentile(vals,95)),1.0); v=np.clip(vals/ceiling,0,1)
            colors=np.column_stack((255*v,255*(1-np.abs(2*v-1)),255*(1-v))).astype(np.uint8); heat_rgb=np.full((h,w,3),255,np.uint8); heat_rgb[gt_mask]=[30,30,30]; heat_rgb[yi[keep],xi[keep]]=colors
            heat=Image.fromarray(heat_rgb,'RGB')
        except ImportError:
            heat=Image.fromarray(rgb,'RGB')
        heat_path=str(Path(p).with_name(Path(p).stem+'_heatmap.png')); heat.save(heat_path)
        messagebox.showinfo('Сохранено',f'Наложение: {p}\nТепловая карта: {heat_path}')

    def _draw_overlay(self,r,t):
        if not hasattr(self,'overlay'): return
        c=self.overlay;c.delete('all'); iw,ih=self.images[1].size; yy,xx=np.where(object_mask(self.images[0])); xy=np.column_stack((xx,yy)).astype(float)@r.T+t
        lo=np.minimum([0,0],xy.min(0)); hi=np.maximum([iw,ih],xy.max(0)); span=np.maximum(hi-lo,1); s=min((c.winfo_width()-16)/span[0],(c.winfo_height()-16)/span[1]); offset=8-lo*s
        base=self.images[1].resize((max(1,int(iw*s)),max(1,int(ih*s))),Image.Resampling.LANCZOS); self.ov_gt=ImageTk.PhotoImage(base); c.create_image(offset[0],offset[1],anchor='nw',image=self.ov_gt)
        xi=np.rint(xy[:,0]*s+offset[0]).astype(int); yi=np.rint(xy[:,1]*s+offset[1]).astype(int)
        keep=(xi>=0)&(yi>=0)&(xi<c.winfo_width())&(yi<c.winfo_height()); layer=np.zeros((max(1,c.winfo_height()),max(1,c.winfo_width()),4),dtype=np.uint8)
        layer[yi[keep],xi[keep],:3]=[220,50,45]; layer[yi[keep],xi[keep],3]=150
        self.ov_map=ImageTk.PhotoImage(Image.fromarray(layer,'RGBA')); c.create_image(0,0,anchor='nw',image=self.ov_map)
        for j,(p,q) in enumerate(zip(self.points[0],self.points[1]),1):
            x,y=np.asarray(q)*s+offset; xp,yp=(np.asarray(p)@r.T+t)*s+offset; col=ACCENTS[(j-1)%len(ACCENTS)]; c.create_line(xp,yp,x,y,fill=col,width=2,dash=(5,3)); c.create_oval(x-7,y-7,x+7,y+7,fill=col,outline='white',width=2);c.create_text(x+10,y-9,text=str(j),fill=INK,anchor='sw',font=FONT_BOLD)

    def _restore(self,bar):
        bar.destroy(); self.overlay.destroy(); del self.overlay; del self.overlay_bar
        self.stage.pack(fill='both', expand=True); self.listbox.pack(fill='x',padx=8); self._show_points_view()

    def _save(self):
        if not hasattr(self,'_last'): messagebox.showinfo('Нет результата','Сначала выполните оценку.'); return
        p=self._file_dialog('save', 'Сохранить связи', Path.cwd(), 'correspondences.json', '.json');
        if p: Path(p).write_text(json.dumps(self._last,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--map',default='/home/optimus/cartographer-map/cartographer_objects_aligned.png');ap.add_argument('--gt',default='/home/optimus/cartographer-map/warehouse_gt_objects.png');a=ap.parse_args();MapAccuracyGUI(a.map,a.gt).mainloop()

if __name__=='__main__': main()
