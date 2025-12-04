import cv2
import deeplake
import os


token="eyJhbGciOiJIUzUxMiIsImlhdCI6MTc2NDAxOTAyMiwiZXhwIjoxNzk1NTU1MDEzfQ.eyJpZCI6IndhaG5hYmVlcyIsIm9yZ19pZCI6IndhaG5hYmVlcyJ9.qWYUgiaR-WiD_FfUcsTVBMhO0pmT8Jt_aiwS5Nsn_3pkqrs6fWbKtts9_dGKSlyQKzZOp1pQ1I3PSySYuFrj7A"

ds=deeplake.load("hub://activeloop/kth-actions",token=token)

save_root="kth_local_avi"
os.makedirs(save_root,exist_ok=True)

for i in range(len(ds)):
    frames=ds["videos"][i].numpy()

    action=ds["action"][i].numpy().item()
    scenario=ds["scenario"][i].numpy().item()

    if isinstance(action,bytes):
        action=action.decode()
    else:
        action=str(action)

    if isinstance(scenario,bytes):
        scenario=scenario.decode()
    else:
        scenario=str(scenario)

    T,H,W,C=frames.shape
    assert C==3,f"Expected 3 channels, got {C}"

    fname=f"clip{i:05d}_{action}_{scenario}.avi"
    out_path=os.path.join(save_root,fname)

    fourcc=cv2.VideoWriter_fourcc(*"MJPG")
    fps=25

    writer=cv2.VideoWriter(out_path,fourcc,fps,(W,H),True)

    for t in range(T):
        frame=frames[t]
        writer.write(frame)

    writer.release()

print("Export complete to",save_root)
