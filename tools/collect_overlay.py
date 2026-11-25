"""Save unreviewed overlay proposals, outside the training dataset.

Offline: python tools/collect_overlay.py --input screenshot.png
Live: python tools/collect_overlay.py --live --every 2 --max 150
Use --all-frames to keep frames even when they have no box proposals.
Use --live --manual to capture only when X is pressed.
Use --live --guided to collect small or uncertain YOLO detections, timed samples, and X snapshots.
Use --make-cleaned to create experimental inpainted copies.
Team assignment is manual; --class-id applies to ALL green boxes in a frame.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid
import cv2
from overlay_boxes import find_boxes, stable_boxes, erase_proposals, yolo_lines

ROOT=Path(__file__).resolve().parents[1]


def play_snapshot_sound():
    import winsound

    sound = ROOT / "assets" / "snapshot.wav"
    try:
        winsound.PlaySound(str(sound), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
    except RuntimeError as exc:
        print(f"[SOUND] Snapshot saved, but sound could not play: {exc}", flush=True)


class KeyEdges:
    """A held key at startup is not a new press."""
    def __init__(self, initial):
        self.previous=dict(initial)

    def update(self, current):
        pressed={key for key,value in current.items() if value and not self.previous.get(key,False)}
        self.previous=dict(current)
        return pressed


def window_state(hwnd, foreground, is_window):
    if not hwnd or not is_window(hwnd):
        return 'window_missing'
    return 'active' if foreground==hwnd else 'focus_paused'


class ScreenCapture:
    def __init__(self, backend):
        self.cam=None
        self.mss=None
        if backend!='mss':
            try:
                import dxcam
                self.cam=dxcam.create(output_idx=0,output_color='BGR')
            except Exception:
                if backend=='dxcam':
                    raise
        if self.cam is None:
            import mss
            factory=getattr(mss,'MSS',None) or mss.mss
            self.mss=factory()
        print('[CAPTURE]', 'dxcam' if self.cam is not None else 'mss',flush=True)

    def grab(self,region=None):
        if self.cam is not None:
            frame=self.cam.grab()
            if frame is None or region is None:
                return frame
            # DXCam output_idx=0 may not be the primary monitor. Use MSS when
            # window-relative physical coordinates cannot be mapped reliably.
            import win32api
            monitor=win32api.GetMonitorInfo(win32api.MonitorFromPoint((0,0)))['Monitor']
            left=region['left']-monitor[0]
            top=region['top']-monitor[1]
            right=left+region['width']
            bottom=top+region['height']
            if left<0 or top<0 or right>frame.shape[1] or bottom>frame.shape[0]:
                raise ValueError('Window outside capture monitor; use --backend mss')
            return frame[top:bottom,left:right].copy()
        import numpy as np
        return cv2.cvtColor(np.asarray(self.mss.grab(region or self.mss.monitors[1])),cv2.COLOR_BGRA2BGR)

    def close(self):
        if self.cam is not None:
            self.cam.release()
        if self.mss is not None:
            self.mss.close()


def client_region(hwnd,gui):
    x1,y1,x2,y2=gui.GetClientRect(hwnd)
    left,top=gui.ClientToScreen(hwnd,(x1,y1))
    right,bottom=gui.ClientToScreen(hwnd,(x2,y2))
    if right<=left or bottom<=top:
        return None
    return dict(left=left,top=top,width=right-left,height=bottom-top)


def review_candidates(current, previous, require_stable=False):
    """Temporal confirmation is metadata by default, not a gate for drafts."""
    stable=stable_boxes(current,previous)
    return [dict(box, temporal_confirmed=box in stable) for box in current
            if not require_stable or box in stable]


def run_live(a,writer):
    import win32api
    import win32gui

    guidance = None
    if getattr(a, "guided", False):
        from guided_capture import GuidedCapture, capture_reason

        guidance = GuidedCapture(ROOT)
    # Keep client coordinates and captured pixels consistent on scaled Windows.
    import ctypes
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError,OSError):
        pass
    keycodes={'stop':{'F2':0x71,'F10':0x79,'F12':0x7B}[a.stop_key], 'bind':0x77,'pause':0x78}
    if a.manual or guidance is not None:
        keycodes['snapshot']=0x58  # X
    def keys():
        return {name:bool(win32api.GetAsyncKeyState(vk)&0x8000) for name,vk in keycodes.items()}
    # Initialise capture BEFORE selecting the game window. Driver startup can
    # take time, and the foreground handle can be transient during Alt+Tab.
    capture=ScreenCapture(a.backend)
    reason='unknown'
    try:
        for i in range(5,0,-1):
            print('Switch to game:',i,flush=True)
            time.sleep(1)
        hwnd=win32gui.GetForegroundWindow()
        title=win32gui.GetWindowText(hwnd) if hwnd and win32gui.IsWindow(hwnd) else '(no window)'
        print(f'[WINDOW] {hwnd}: {title}',flush=True)
        print(f'[KEYS] {a.stop_key}=stop; F8=select current game window; F9=pause',flush=True)
        if guidance is not None:
            print(f'[SAVE] YOLO checks every {a.every:g}s; periodic sample every {a.sample_every:g}s; X=snapshot with sound', flush=True)
        elif a.manual:
            print('[SAVE] X=snapshot with sound confirmation; automatic capture is off',flush=True)
        else:
            print('[SAVE] '+('Two-frame confirmation required' if a.require_stable else 'All geometric candidates go to manual review; temporal match is optional'),flush=True)
        edges=KeyEdges(keys())
        previous=[]
        paused=False
        last_save=-float('inf')
        next_yolo_check = 0.0
        next_periodic_sample = time.monotonic() + a.sample_every if guidance is not None else float('inf')
        last_report=-float('inf')
        frames=empty=green=red=confirmed=green_frames=red_frames=0
        debug_saved=False
        last_box_debug=-float('inf')
        while writer.count<a.max:
            pressed=edges.update(keys())
            if 'stop' in pressed:
                reason=f'{a.stop_key} pressed'
                break
            if 'bind' in pressed:
                candidate=win32gui.GetForegroundWindow()
                if candidate and win32gui.IsWindow(candidate):
                    hwnd=candidate
                    previous=[]
                    print(f'[WINDOW] selected {hwnd}: {win32gui.GetWindowText(hwnd)}',flush=True)
                else:
                    print('[WINDOW] No valid foreground window. Switch to game and press F8.',flush=True)
            if 'pause' in pressed:
                paused=not paused
                previous=[]
                print('[PAUSE]',paused,flush=True)
            state='manual_pause' if paused else window_state(hwnd,win32gui.GetForegroundWindow(),win32gui.IsWindow)
            now=time.monotonic()
            if now-last_report>=5:
                if guidance is not None:
                    print(f'[STATUS] {state}; frames={frames}; no_new_frame={empty}; saved={writer.count}', flush=True)
                else:
                    print(f'[STATUS] {state}; frames={frames}; no_new_frame={empty}; green={green}; red={red}; confirmed_green={confirmed}; green_frames={green_frames}; red_frames={red_frames}; saved={writer.count}',flush=True)
                if state=='window_missing':
                    print('[WAIT] Window unavailable. Switch to game and press F8. Collector is still running.',flush=True)
                elif state=='focus_paused':
                    print('[WAIT] Return to selected game, or press F8 in the correct game window.',flush=True)
                last_report=now
            if state!='active':
                previous=[]
                time.sleep(.1)
                continue
            snapshot = 'snapshot' in pressed
            periodic = now >= next_periodic_sample
            if not snapshot and (a.manual or (guidance is not None and now < next_yolo_check and not periodic)):
                time.sleep(.02)
                continue
            region=client_region(hwnd,win32gui)
            if region is None:
                previous=[]
                time.sleep(.1)
                continue
            frame=capture.grab(region)
            if frame is None:
                empty+=1
                if snapshot:
                    print('[SKIP] No new image from capture; press X again',flush=True)
            elif guidance is not None:
                frames += 1
                proposals = []
                trigger = "manual"
                if not snapshot:
                    next_yolo_check = now + a.every
                    proposals = guidance.propose(frame)
                    trigger = capture_reason(proposals, frame.shape[0], periodic)
                    if periodic:
                        next_periodic_sample = now + a.sample_every
                if trigger:
                    details = {"reason": trigger, "weights": guidance.weights.name, "proposals": proposals}
                    saved = writer.save(frame, [], 'live_game_client', guidance=details)
                    if snapshot:
                        if saved:
                            play_snapshot_sound()
                        else:
                            print('[SKIP] Identical frame already saved', flush=True)
            else:
                frames+=1
                current=find_boxes(frame)
                stable=stable_boxes(current,previous)
                candidates=review_candidates(current,previous,a.require_stable)
                previous=current
                green=sum(b['color']=='green' for b in current)
                red=sum(b['color']=='red' for b in current)
                confirmed=sum(b['color']=='green' for b in stable)
                green_frames+=int(green>0)
                red_frames+=int(red>0)
                box_debug=bool(current) and now-last_box_debug>=5
                if a.debug and (not debug_saved or now-last_report<.3 or box_debug):
                    target=writer.out/'debug'
                    target.mkdir(exist_ok=True)
                    ok,data=cv2.imencode('.png',frame)
                    if not ok: raise RuntimeError('Debug PNG encoding failed')
                    (target/'latest_capture.png').write_bytes(data.tobytes())
                    (target/'latest_boxes.json').write_text(json.dumps(current,indent=2),encoding='utf-8')
                    if box_debug:
                        (target/'latest_with_boxes.png').write_bytes(data.tobytes())
                        (target/'latest_with_boxes.json').write_text(json.dumps(current,indent=2),encoding='utf-8')
                        last_box_debug=now
                    if not debug_saved:
                        print('[DEBUG] Captured view:',target/'latest_capture.png',flush=True)
                    debug_saved=True
                if a.manual or now-last_save>=a.every:
                    if writer.save(frame,candidates,'live_game_client'):
                        last_save=now
                        if a.manual:
                            play_snapshot_sound()
                    elif a.manual:
                        print('[SKIP] Identical frame already saved',flush=True)
            time.sleep(.02 if a.manual or guidance is not None else .25)
        else:
            reason=f'limit reached ({a.max})'
    except KeyboardInterrupt:
        reason='Ctrl+C'
    except Exception as exc:
        reason=f'error: {type(exc).__name__}: {exc}'
        raise
    finally:
        print('[STOP]',reason,flush=True)
        capture.close()


class Writer:
    def __init__(self,out,make_cleaned=False,class_id=None,keep_red=False,*,all_frames=False):
        self.out=Path(out)
        self.out.mkdir(parents=True,exist_ok=False)
        self.make_cleaned=make_cleaned
        self.class_id=class_id
        self.keep_red=keep_red
        self.all_frames=all_frames
        self.seen=set()
        self.count=0

    def save(self,frame,boxes,source,*,guidance=None):
        if not self.all_frames:
            if not boxes or (not self.keep_red and not any(b['color']=='green' for b in boxes)):
                return False
        digest=hashlib.sha256(frame.tobytes()).hexdigest()
        if digest in self.seen:
            return False
        self.seen.add(digest)
        self.count+=1
        stem=f'frame_{self.count:05d}'
        def write(folder,image):
            target=self.out/folder
            target.mkdir(exist_ok=True)
            ok,data=cv2.imencode('.png',image)
            if not ok:
                raise RuntimeError('PNG encoding failed')
            (target/(stem+'.png')).write_bytes(data.tobytes())
        write('originals',frame)
        record={'source':source,'review_required':True,'image_size':[frame.shape[1],frame.shape[0]],
                'boxes':boxes,'assigned_class_for_all_green':self.class_id,
                'warning':'Color indicates visibility only; team unknown. Red boxes must not label invisible enemies.'}
        if guidance is not None:
            record["guided_capture"] = guidance
            record["warning"] = "YOLO proposals require manual review before training."
        (self.out/(stem+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
        if self.make_cleaned:
            # Remove every detected outline, including unstable detections and
            # red outlines. Missed outlines and names can still remain.
            clean,mask=erase_proposals(frame,find_boxes(frame))
            write('experimental_cleaned',clean)
            write('erase_masks',mask)
        if self.class_id is not None:
            labels=yolo_lines(boxes,frame.shape,self.class_id)
            if labels:
                target=self.out/'draft_labels'
                target.mkdir(exist_ok=True)
                (target/(stem+'.txt')).write_text(labels,encoding='utf-8')
        if guidance is not None:
            print(f'[GUIDED] {stem}: {guidance["reason"]}; {len(guidance["proposals"])} YOLO proposals', flush=True)
        else:
            print(f'[REVIEW] {stem}: {len(boxes)} boxes',flush=True)
        return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path,help='PNG/JPG file or directory')
    source.add_argument('--live',action='store_true')
    ap.add_argument('--out',type=Path)
    ap.add_argument('--every',type=float,default=2)
    ap.add_argument('--max',type=int,default=150)
    ap.add_argument('--make-cleaned',action='store_true')
    capture_mode=ap.add_mutually_exclusive_group()
    capture_mode.add_argument('--manual',action='store_true',help='Live only: X takes a snapshot, including frames without colored boxes')
    capture_mode.add_argument('--guided',action='store_true',help='Live only: use YOLO to select frames, with periodic and X snapshots')
    ap.add_argument('--sample-every',type=float,default=20,help='Seconds between periodic snapshots in guided mode')
    ap.add_argument('--all-frames',action='store_true',help='Keep frames even when no colored boxes are detected')
    ap.add_argument('--keep-red',action='store_true',help='Keep red-only frames for review, never label them')
    ap.add_argument('--class-id',type=int,choices=[0,1],help='MANUAL assumption for ALL green boxes: 0 enemy, 1 teammate')
    ap.add_argument('--backend',choices=['auto','dxcam','mss'],default='auto')
    ap.add_argument('--stop-key',choices=['F2','F10','F12'],default='F10')
    ap.add_argument('--debug',action='store_true',help='Save latest captured screen and detected boxes even when no proposals are saved')
    ap.add_argument('--require-stable',action='store_true',help='Optional strict mode: save only boxes matched in two consecutive frames')
    a=ap.parse_args()
    if not .25 <= a.every < float('inf') or a.max <= 0:
        ap.error('--every >= 0.25; --max > 0')
    if a.manual and not a.live:
        ap.error('--manual requires --live')
    if a.guided:
        if not a.live:
            ap.error('--guided requires --live')
        if not .25 <= a.sample_every < float('inf'):
            ap.error('--sample-every must be a finite number >= 0.25')
        if a.class_id is not None or a.make_cleaned or a.all_frames or a.keep_red or a.require_stable or a.debug:
            ap.error('--guided collects raw frames with YOLO proposals; use overlay-specific options without --guided')
    if a.input and not a.input.exists():
        ap.error('Input does not exist')
    out=a.out or ROOT/'overlay_review'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    writer=Writer(out,a.make_cleaned,a.class_id,a.keep_red,all_frames=a.all_frames or a.manual or a.guided)
    if a.all_frames and not a.manual:
        print('[SAVE] Keeping frames even without box proposals',flush=True)
    print('UNREVIEWED proposals only. Output:',out,flush=True)
    if a.input:
        import numpy as np
        files=[a.input] if a.input.is_file() else sorted(a.input.rglob('*'))
        for path in files:
            if path.suffix.lower() not in ('.png','.jpg','.jpeg','.bmp'):
                continue
            frame=cv2.imdecode(np.frombuffer(path.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
            if frame is None:
                print('Unreadable:',path)
                continue
            writer.save(frame,find_boxes(frame),str(path))
            if writer.count>=a.max:
                break
    else:
        run_live(a,writer)
    print('Saved:',writer.count,'; folder:',writer.out)


if __name__=='__main__':
    main()
